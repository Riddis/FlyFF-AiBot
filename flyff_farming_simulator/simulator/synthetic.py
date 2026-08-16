from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .environment import RecordedFarmingEnv
from .map_model import MapModel
from .movement_kernel import MOVEMENT_PHYSICS_MODEL_ID, STEADY_TURN_RADIANS, SteeringDirection, advance_player_tick
from .world_model import MovementModel, RecordedWorldModel

try:  # Optional until training is requested.
    import gymnasium as gym

    _BaseEnv = gym.Env
except ImportError:  # pragma: no cover
    gym = None

    class _BaseEnv:  # type: ignore[no-redef]
        pass


CURRICULUM_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SyntheticVariantSpec:
    name: str
    stage: str
    template: str
    density_profile: str
    respawn_profile: str
    map_assets: str
    world_model: str
    weight: float
    seed: int
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SyntheticCurriculum:
    schema_version: int
    name: str
    generated_seed: int
    variants: tuple[SyntheticVariantSpec, ...]
    design_rules: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticCurriculum":
        manifest = Path(path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        variants = tuple(SyntheticVariantSpec(**item) for item in payload["variants"])
        return cls(
            schema_version=int(payload["schema_version"]),
            name=str(payload["name"]),
            generated_seed=int(payload["generated_seed"]),
            variants=variants,
            design_rules=tuple(str(value) for value in payload.get("design_rules", ())),
        )

    def save(self, path: str | Path) -> Path:
        manifest = Path(path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "name": self.name,
                    "generated_seed": self.generated_seed,
                    "variants": [asdict(item) for item in self.variants],
                    "design_rules": list(self.design_rules),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def select(self, stage: str = "all") -> tuple[SyntheticVariantSpec, ...]:
        selected = self.variants if stage == "all" else tuple(
            item for item in self.variants if item.stage == stage
        )
        if not selected:
            raise ValueError(f"Curriculum contains no variants for stage {stage!r}")
        return selected


def _ellipse_mask(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    angle: float = 0.0,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx - center_x
    dy = yy - center_y
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_x = dx * cos_a + dy * sin_a
    rotated_y = -dx * sin_a + dy * cos_a
    return (rotated_x / max(1.0, radius_x)) ** 2 + (rotated_y / max(1.0, radius_y)) ** 2 <= 1.0


def _disk_mask(shape: tuple[int, int], center_x: float, center_y: float, radius: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius * radius


def _largest_component(mask: np.ndarray) -> np.ndarray:
    walkable = np.asarray(mask, dtype=bool)
    height, width = walkable.shape
    seen = np.zeros_like(walkable, dtype=bool)
    best: list[tuple[int, int]] = []
    for start_y, start_x in np.argwhere(walkable):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_x), int(start_y))]
        seen[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height and walkable[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(walkable, dtype=bool)
    for x, y in best:
        result[y, x] = True
    return result


def _nearest_safe_cell(map_model: MapModel, desired: tuple[float, float]) -> tuple[int, int]:
    safe = np.argwhere(map_model.features.safe_traversable)
    if len(safe) == 0:
        safe = np.argwhere(map_model.traversable)
    target_x, target_y = desired
    distances = (safe[:, 1] - target_x) ** 2 + (safe[:, 0] - target_y) ** 2
    index = int(np.argmin(distances))
    y, x = safe[index]
    return int(x), int(y)


def _place_obstacles(
    traversable: np.ndarray,
    rng: np.random.Generator,
    *,
    obstacle_count: int,
    spawn_hint: tuple[float, float],
    partial_walls: int,
) -> np.ndarray:
    result = np.asarray(traversable, dtype=bool).copy()
    height, width = result.shape
    spawn_x, spawn_y = spawn_hint
    free_points = np.argwhere(result)
    for _ in range(obstacle_count):
        if len(free_points) == 0:
            break
        for _attempt in range(100):
            y, x = free_points[int(rng.integers(0, len(free_points)))]
            radius_x = float(rng.uniform(2.5, 7.5))
            radius_y = float(rng.uniform(2.5, 7.5))
            if math.hypot(float(x) - spawn_x, float(y) - spawn_y) < 18.0:
                continue
            obstacle = _ellipse_mask(
                result.shape,
                float(x),
                float(y),
                radius_x,
                radius_y,
                float(rng.uniform(-math.pi, math.pi)),
            )
            candidate = result & ~obstacle
            if np.count_nonzero(candidate) >= np.count_nonzero(result) * 0.985:
                result = candidate
                break

    # Partial walls split open areas without creating corridor/dungeon navigation.
    # Each wall has one or two deliberately wide gaps.
    for _ in range(partial_walls):
        horizontal = bool(rng.integers(0, 2))
        thickness = int(rng.integers(3, 6))
        if horizontal:
            y = int(rng.integers(height * 3 // 10, height * 7 // 10))
            x0 = int(rng.integers(width // 8, width // 3))
            x1 = int(rng.integers(width * 2 // 3, width * 7 // 8))
            wall = np.zeros_like(result)
            wall[max(0, y - thickness // 2) : min(height, y + thickness // 2 + 1), x0:x1] = True
            gap_width = int(rng.integers(22, 38))
            gap_center = int(rng.integers(x0 + gap_width // 2, x1 - gap_width // 2))
            wall[:, gap_center - gap_width // 2 : gap_center + gap_width // 2] = False
        else:
            x = int(rng.integers(width * 3 // 10, width * 7 // 10))
            y0 = int(rng.integers(height // 8, height // 3))
            y1 = int(rng.integers(height * 2 // 3, height * 7 // 8))
            wall = np.zeros_like(result)
            wall[y0:y1, max(0, x - thickness // 2) : min(width, x + thickness // 2 + 1)] = True
            gap_width = int(rng.integers(22, 38))
            gap_center = int(rng.integers(y0 + gap_width // 2, y1 - gap_width // 2))
            wall[gap_center - gap_width // 2 : gap_center + gap_width // 2, :] = False
        candidate = result & ~wall
        connected = _largest_component(candidate)
        if np.count_nonzero(connected) >= np.count_nonzero(candidate) * 0.985:
            result = connected
    return _largest_component(result)


# Escapability validation ----------------------------------------------------
#
# A "stuck" state is one where the bot's real latched-forward controls
# (STRAIGHT/LEFT/RIGHT, with the same sliding collision response
# ``RecordedFarmingEnv`` uses live) cannot make meaningful progress. Left
# unchecked, obstacle placement can create concave pockets or dead ends the
# bot wanders into and never leaves -- corrupting BC/teacher datasets with
# degenerate repeated-label episodes and letting a PPO rollout waste an
# entire episode wedged against a wall. Each stage gets a progressively
# larger tick budget to escape within, but every stage requires a proof of
# escapability; none accepts a truly unescapable state.
#
# Recalibrated after obstacle_radius_cells -> 0 (map_model.py): boundary
# positions are now sampled flush against the real wall/corner instead of
# ~2 cells back (the old OBSTACLE_BUFFER margin), so turning away from a
# tight corner genuinely costs more ticks even though nothing is actually
# unescapable.
#
# These are NOT chosen merely to make every generated layout pass -- that
# would let the gate quietly stop filtering out layouts that are too hard
# for a given stage. Two things were checked instead (2026-08-07 diagnostics):
#   1. What one tick actually IS: at the default movement reference, 1 tick
#      = 0.2s of simulated time, a LEFT/RIGHT tap turns ~5.73 degrees, and a
#      STRAIGHT tap covers ~2.2 cells. Traced the actual winning action
#      sequence for several worst-case boundary positions (found by
#      exhaustively searching up to a 40-tick ceiling, independent of any
#      candidate budget): the ACTUAL minimum ticks needed was 21-23, i.e.
#      what does not fit in the old 8-tick budget -- not "8 ticks looks like
#      this". Those 21-23 ticks are one sustained, monotonic turn in a
#      single direction (zero direction reversals, 1-2 short forward taps
#      mixed in) totaling ~110-120 degrees over ~4.2-4.6s: a single
#      corrective turn at a bad approach angle to an ordinary corner, not a
#      multi-step wedge/pocket recovery maneuver. That 21-23 tick reality is
#      what early=24 was sized against, with a small margin.
#   2. Whether a SMALLER budget, combined with the existing
#      regenerate-on-failure loop (_generate_validated_layout, up to
#      _MAX_LAYOUT_ATTEMPTS attempts), already succeeds -- i.e. whether the
#      gate can keep doing its real job (reject/regenerate away from harder
#      layouts) at a smaller number instead of being raised until it accepts
#      everything. 12 and 16 ticks FAIL for every early template even after
#      exhausting all 40 attempts (every random early layout contains some
#      boundary position needing more); 20-24 ticks succeed for every
#      template within 1-2 attempts. Same check for intermediate/advanced
#      (their own, deliberately harder, obstacle_level settings): both
#      already succeed on the first attempt from a modest budget.
#
# early=24 (4.8s, ~one 110-120 degree corrective turn) keeps the gate doing
# real filtering (occasional regeneration, not automatic acceptance) at
# close to the smallest value that works at all. intermediate=32 (6.4s,
# enough for just over a full 180 degree reversal) and advanced=40 (8.0s, a
# full reversal plus continued correction) step up deliberately, matching
# those stages' own harder obstacle_level geometry, not padding chosen to
# avoid regeneration.
#
# IMPORTANT semantic note: this gate is a MAP-SANITY constraint, not a
# curriculum/graduation skill target. It only proves the movement system
# COULD recover from a sampled wall-adjacent position within a stage's
# budget if it ever ended up there -- it exists to reject pathologically
# unforgiving geometry (concave pockets, dead ends) from ever being
# generated, nothing more. It does not mean early/Beginner is meant to
# teach or tolerate 24-tick wedge recovery as normal behavior. A graduated
# Beginner policy should simply not be getting into these states at all;
# see milestone_evaluator's contact-rate criteria for that separate bar.
_STAGE_ESCAPE_TICKS: dict[str, int] = {"early": 24, "intermediate": 32, "advanced": 40}
_ESCAPE_MAX_VISITED_STATES = 4_000
_ESCAPE_SAMPLE_POSITIONS = 120
_ESCAPE_HEADINGS_PER_POSITION = 4


def _regains_movement_within(
    map_model: MapModel,
    x: float,
    z: float,
    heading: float,
    *,
    max_ticks: int,
    previous_steering: SteeringDirection = SteeringDirection.NONE,
) -> bool:
    """Whether the bot's real controls can regain a genuinely clear
    direction -- a full ordinary movement step with no contact at all --
    within ``max_ticks`` control ticks, branching over STRAIGHT/LEFT/RIGHT.

    Success requires an uncontacted step, not merely any displacement: a
    sliding move that is still blocked can shuffle the player sideways, or
    even deeper into a dead end, while never actually freeing them, and
    counting that as "escaped" would pass layouts that still trap the bot.
    Continued search between ticks uses the kernel's own substep/slide
    response so the frontier reflects where the bot's controls would
    actually carry it while still blocked, matching
    ``RecordedFarmingEnv``'s live collision handling.

    2026-08-13: migrated from the legacy per-action Gaussian model's fixed
    turn/distance stats to movement_kernel.advance_player_tick -- the same
    authoritative kernel RecordedFarmingEnv/the planner/the oracle all use,
    so this validation gate (which determines whether a generated layout
    is shippable) reflects what the real simulator now does, not the old
    ~10-15deg/tick model. previous_steering is threaded through the
    frontier since the kernel's turn is stateful (onset vs. steady); each
    call starts fresh (NONE) by default, matching a probed position/
    heading with no prior steering commitment (both call sites below probe
    positions independently, never mid-sequence)."""

    unit = max(1.0e-6, map_model.native_units_per_cell)
    turn_step = STEADY_TURN_RADIANS  # discretization bucket size only

    def discretize(px: float, pz: float, heading_value: float) -> tuple[int, int, int]:
        return (
            int(round(px / unit * 2.0)),
            int(round(pz / unit * 2.0)),
            int(round(heading_value / turn_step)),
        )

    frontier: list[tuple[float, float, float, SteeringDirection]] = [(x, z, heading, previous_steering)]
    visited = {discretize(x, z, heading)}
    for _tick in range(int(max_ticks)):
        next_frontier: list[tuple[float, float, float, SteeringDirection]] = []
        for px, pz, ph, pprev in frontier:
            for direction in (SteeringDirection.NONE, SteeringDirection.LEFT, SteeringDirection.RIGHT):
                result = advance_player_tick(map_model, px, pz, ph, pprev, direction)
                if not result.contact:
                    return True
                key = discretize(result.x, result.z, result.heading)
                if key in visited:
                    continue
                visited.add(key)
                if len(visited) > _ESCAPE_MAX_VISITED_STATES:
                    return False
                next_frontier.append((result.x, result.z, result.heading, result.next_previous_steering))
        if not next_frontier:
            return False
        frontier = next_frontier
    return False


def _boundary_safe_positions(map_model: MapModel) -> np.ndarray:
    """Safe cells adjacent to any non-safe neighbor (obstacle, its buffer,
    or the map edge) -- exactly where a concave pocket or dead end could
    exist. Interior open cells can never be enclosed and would waste
    validation time, so they are excluded from sampling entirely.
    """

    safe = np.asarray(map_model.features.safe_traversable, dtype=bool)
    not_safe = ~safe
    padded = np.pad(not_safe, 1, mode="constant", constant_values=True)
    boundary = np.zeros_like(safe)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            boundary |= padded[1 + dy : 1 + dy + safe.shape[0], 1 + dx : 1 + dx + safe.shape[1]]
    boundary &= safe
    return np.argwhere(boundary)


def _layout_escapability_reasons(
    map_model: MapModel,
    rng: np.random.Generator,
    *,
    stage: str,
    spawn_native: tuple[float, float],
) -> list[str]:
    """Sample obstacle-adjacent positions and headings and require every one
    to regain meaningful movement within the stage's tick budget. The spawn
    point is always additionally required to pass the strictest (early-stage)
    budget, so a player never starts wedged against an obstacle corner
    regardless of how lenient the surrounding stage is.
    """

    max_ticks = _STAGE_ESCAPE_TICKS[stage]
    reasons: list[str] = []

    # RecordedFarmingEnv.reset() assigns a uniform-random heading at spawn
    # (``self.heading = rng.uniform(-pi, pi)``), so the spawn position must
    # be escapable from every sampled heading, not merely some of them.
    spawn_ticks = _STAGE_ESCAPE_TICKS["early"]
    failing_headings = [
        heading
        for heading in np.linspace(-math.pi, math.pi, 8, endpoint=False)
        if not _regains_movement_within(
            map_model, spawn_native[0], spawn_native[1], heading, max_ticks=spawn_ticks
        )
    ]
    if failing_headings:
        reasons.append(
            f"spawn point is not escapable from {len(failing_headings)}/8 sampled headings "
            "within the turning envelope"
        )

    boundary_cells = _boundary_safe_positions(map_model)
    if len(boundary_cells) == 0:
        return reasons
    sample_count = min(int(_ESCAPE_SAMPLE_POSITIONS), len(boundary_cells))
    chosen = boundary_cells[rng.choice(len(boundary_cells), size=sample_count, replace=False)]
    failures = 0
    first_failure: tuple[int, int] | None = None
    for cell_y, cell_x in chosen:
        native_x, native_z = map_model.layout_to_native(float(cell_x), float(cell_y))
        headings = rng.uniform(-math.pi, math.pi, size=_ESCAPE_HEADINGS_PER_POSITION)
        for heading in headings:
            if not _regains_movement_within(
                map_model, native_x, native_z, float(heading), max_ticks=max_ticks
            ):
                failures += 1
                if first_failure is None:
                    first_failure = (int(cell_x), int(cell_y))
                break
    if failures:
        reasons.append(
            f"{failures}/{sample_count} sampled obstacle-adjacent positions cannot regain movement "
            f"within {max_ticks} ticks (first failure near layout cell {first_failure})"
        )
    return reasons


def _generate_layout(
    template: str,
    rng: np.random.Generator,
    *,
    size: int,
    obstacle_level: int,
) -> tuple[MapModel, tuple[int, int], dict[str, object]]:
    shape = (size, size)
    center = (size - 1) / 2.0
    if template == "open_field":
        traversable = _ellipse_mask(
            shape,
            center,
            center,
            size * 0.43,
            size * 0.37,
            float(rng.uniform(-0.20, 0.20)),
        )
        partial_walls = 0
    elif template == "irregular_plain":
        traversable = _ellipse_mask(shape, center, center, size * 0.37, size * 0.34)
        for angle in np.linspace(0.0, 2.0 * math.pi, 5, endpoint=False):
            radius = size * 0.18
            traversable |= _ellipse_mask(
                shape,
                center + math.cos(angle) * size * 0.23,
                center + math.sin(angle) * size * 0.20,
                radius * float(rng.uniform(0.9, 1.2)),
                radius * float(rng.uniform(0.8, 1.2)),
                float(rng.uniform(-0.5, 0.5)),
            )
        partial_walls = 0
    elif template == "broad_lobes":
        traversable = _disk_mask(shape, center, center, size * 0.22)
        lobe_count = int(rng.integers(3, 6))
        for index in range(lobe_count):
            angle = 2.0 * math.pi * index / lobe_count + float(rng.uniform(-0.22, 0.22))
            distance = size * float(rng.uniform(0.20, 0.29))
            traversable |= _ellipse_mask(
                shape,
                center + math.cos(angle) * distance,
                center + math.sin(angle) * distance,
                size * float(rng.uniform(0.18, 0.24)),
                size * float(rng.uniform(0.16, 0.22)),
                angle + float(rng.uniform(-0.35, 0.35)),
            )
        partial_walls = 0
    elif template == "wide_neck":
        offset = size * 0.22
        left = _ellipse_mask(shape, center - offset, center, size * 0.28, size * 0.33)
        right = _ellipse_mask(shape, center + offset, center, size * 0.28, size * 0.33)
        bridge = np.zeros(shape, dtype=bool)
        half_height = int(size * 0.10)
        bridge[int(center) - half_height : int(center) + half_height + 1, int(center - offset) : int(center + offset) + 1] = True
        traversable = left | right | bridge
        partial_walls = 0
    elif template == "split_field":
        traversable = _ellipse_mask(shape, center, center, size * 0.43, size * 0.38)
        partial_walls = 1
    elif template == "open_center":
        traversable = _ellipse_mask(shape, center, center, size * 0.42, size * 0.40)
        partial_walls = 0
        # Preserve a large clear central arena; obstacles are pushed outward later.
    else:
        raise ValueError(f"Unknown synthetic map template: {template}")

    traversable = _largest_component(traversable)
    spawn_hint = (center, center + size * 0.20)
    obstacle_count = {
        0: int(rng.integers(1, 4)),
        1: int(rng.integers(4, 8)),
        2: int(rng.integers(7, 13)),
    }[int(np.clip(obstacle_level, 0, 2))]
    if template == "open_center":
        obstacle_count += 3
    traversable = _place_obstacles(
        traversable,
        rng,
        obstacle_count=obstacle_count,
        spawn_hint=spawn_hint,
        partial_walls=partial_walls,
    )
    model = MapModel.from_arrays(traversable, native_units_per_cell=1.6)
    spawn = _nearest_safe_cell(model, spawn_hint)
    free_fraction = float(np.count_nonzero(model.traversable) / model.traversable.size)
    safe_fraction = float(np.count_nonzero(model.features.safe_traversable) / max(1, np.count_nonzero(model.traversable)))
    metadata = {
        "template": template,
        "free_fraction": free_fraction,
        "safe_fraction_of_free": safe_fraction,
        "spawn_cell": [spawn[0], spawn[1]],
        "design": "large_open_farming_area",
        "corridor_generation": False,
    }
    return model, spawn, metadata


def _density_configuration(profile: str, rng: np.random.Generator) -> tuple[int, int, float]:
    if profile == "low":
        return int(rng.integers(280, 421)), int(rng.integers(10, 15)), 0.75
    if profile == "typical":
        return int(rng.integers(480, 721)), int(rng.integers(15, 23)), 1.0
    if profile == "high":
        return int(rng.integers(720, 1001)), int(rng.integers(20, 31)), 1.15
    if profile == "uneven":
        return int(rng.integers(520, 821)), int(rng.integers(14, 23)), 1.35
    if profile == "shifting":
        return int(rng.integers(560, 901)), int(rng.integers(16, 26)), 1.20
    raise ValueError(f"Unknown density profile: {profile}")


def _respawn_samples(profile: str, rng: np.random.Generator, count: int = 2_000) -> tuple[float, ...]:
    if profile == "fast":
        values = rng.normal(7.5, 1.8, size=count)
    elif profile == "typical":
        values = rng.normal(12.0, 3.5, size=count)
    elif profile == "variable":
        values = rng.lognormal(mean=math.log(12.0), sigma=0.45, size=count)
    elif profile == "bursty":
        selector = rng.random(count) < 0.60
        values = np.where(selector, rng.normal(7.0, 1.2, count), rng.normal(17.0, 3.0, count))
    elif profile == "slow":
        values = rng.normal(18.0, 5.0, size=count)
    else:
        raise ValueError(f"Unknown respawn profile: {profile}")
    return tuple(float(value) for value in np.clip(values, 2.0, 35.0))


def _clustered_spawn_reservoirs(
    map_model: MapModel,
    rng: np.random.Generator,
    *,
    section_count: int,
    cluster_count: int,
    density_profile: str,
    samples_per_cluster: int = 180,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    section_total = section_count + 1
    safe_points = np.argwhere(map_model.features.safe_traversable)
    if len(safe_points) == 0:
        safe_points = np.argwhere(map_model.traversable)
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < cluster_count and attempts < cluster_count * 300:
        attempts += 1
        y, x = safe_points[int(rng.integers(0, len(safe_points)))]
        candidate = (int(x), int(y))
        minimum = 9.0 if density_profile == "high" else 13.0
        if all(math.hypot(candidate[0] - cx, candidate[1] - cy) >= minimum for cx, cy in centers):
            centers.append(candidate)
    if not centers:
        y, x = safe_points[0]
        centers.append((int(x), int(y)))

    reservoirs: list[list[tuple[float, float]]] = [[] for _ in range(section_total)]
    for index, (center_x, center_y) in enumerate(centers):
        if density_profile == "uneven":
            sigma = float(rng.uniform(2.0, 7.0))
            sample_count = int(samples_per_cluster * (2.2 if index < max(1, len(centers) // 4) else 0.65))
        elif density_profile == "low":
            sigma = float(rng.uniform(4.0, 8.5))
            sample_count = int(samples_per_cluster * 0.75)
        elif density_profile == "high":
            sigma = float(rng.uniform(2.0, 5.5))
            sample_count = int(samples_per_cluster * 1.25)
        else:
            sigma = float(rng.uniform(3.0, 7.0))
            sample_count = samples_per_cluster
        for _ in range(max(40, sample_count)):
            x = int(round(rng.normal(center_x, sigma)))
            y = int(round(rng.normal(center_y, sigma)))
            if not (0 <= x < map_model.traversable.shape[1] and 0 <= y < map_model.traversable.shape[0]):
                continue
            if not map_model.features.safe_traversable[y, x]:
                continue
            native = map_model.layout_to_native(x, y)
            section = map_model.section(*native, section_count=section_count)
            reservoirs[section].append(native)
    all_positions = [position for reservoir in reservoirs for position in reservoir]
    if not all_positions:
        all_positions = [map_model.layout_to_native(*map_model.random_safe_cell(rng))]
    for section in range(section_total):
        if len(reservoirs[section]) < 40:
            reservoirs[section].extend(all_positions[: min(200, len(all_positions))])
    return tuple(tuple(section) for section in reservoirs)


def _section_probabilities(
    reservoirs: tuple[tuple[tuple[float, float], ...], ...],
    rng: np.random.Generator,
    *,
    density_profile: str,
) -> tuple[float, ...]:
    counts = np.asarray([max(1, len(section)) for section in reservoirs], dtype=np.float64)
    if density_profile == "uneven":
        counts *= rng.lognormal(mean=0.0, sigma=0.8, size=len(counts))
    elif density_profile == "shifting":
        counts *= rng.lognormal(mean=0.0, sigma=0.45, size=len(counts))
    counts /= counts.sum()
    return tuple(float(value) for value in counts)


def _transition_matrix(
    base: tuple[float, ...],
    *,
    density_profile: str,
) -> tuple[tuple[float, ...], ...]:
    probabilities = np.asarray(base, dtype=np.float64)
    rows: list[tuple[float, ...]] = []
    for source in range(len(probabilities)):
        row = probabilities.copy()
        if density_profile == "shifting":
            row[source] *= 0.12
        elif density_profile == "uneven":
            row[source] *= 0.45
        else:
            row[source] *= 0.75
        row /= row.sum()
        rows.append(tuple(float(value) for value in row))
    return tuple(rows)


def _movement_from_reference(reference: RecordedWorldModel | None) -> tuple[MovementModel, ...]:
    if reference is not None:
        return reference.movement
    return (
        MovementModel(0, 2.20, 0.25, 0.0, 0.01),
        MovementModel(0, 1.75, 0.24, 0.10, 0.025),
        MovementModel(0, 1.75, 0.24, -0.10, 0.025),
        MovementModel(0, 0.0, 0.01, 0.0, 0.005),
        MovementModel(0, 2.20, 0.25, 0.0, 0.01),
    )


def _variant_plan(count: int) -> list[tuple[str, str, str, str, int]]:
    templates = ["open_field", "irregular_plain", "broad_lobes", "wide_neck", "split_field", "open_center"]
    stage_profiles = {
        "early": [("typical", "fast", 0), ("high", "typical", 0), ("low", "fast", 0), ("typical", "bursty", 1)],
        "intermediate": [("uneven", "typical", 1), ("typical", "variable", 1), ("high", "bursty", 1), ("shifting", "typical", 1)],
        "advanced": [("shifting", "variable", 2), ("uneven", "slow", 2), ("low", "variable", 2), ("high", "slow", 2)],
    }
    plan: list[tuple[str, str, str, str, int]] = []
    index = 0
    stages = ("early", "intermediate", "advanced")
    while len(plan) < count:
        stage = stages[index % len(stages)]
        profile_list = stage_profiles[stage]
        density, respawn, obstacles = profile_list[(index // len(stages)) % len(profile_list)]
        template = templates[index % len(templates)]
        plan.append((stage, template, density, respawn, obstacles))
        index += 1
    plan.sort(key=lambda item: stages.index(item[0]))
    return plan


_MAX_LAYOUT_ATTEMPTS = 40


def _generate_validated_layout(
    template: str,
    base_seed: int,
    *,
    obstacle_level: int,
    stage: str,
) -> tuple[MapModel, tuple[int, int], dict[str, object], np.random.Generator]:
    """Regenerate a layout until it passes the stage's escapability gate,
    rather than silently shipping one that can trap the bot. Each attempt is
    a full, independently-seeded regeneration (template shape, size, and
    obstacle placement together) so a failing attempt never leaves partial
    state behind for the next one.
    """

    last_reasons: list[str] = []
    for attempt in range(_MAX_LAYOUT_ATTEMPTS):
        attempt_rng = np.random.default_rng(base_seed + attempt * 104_729)
        size = int(attempt_rng.integers(188, 241))
        map_model, spawn_cell, metadata = _generate_layout(
            template, attempt_rng, size=size, obstacle_level=obstacle_level
        )
        spawn_native = map_model.layout_to_native(*spawn_cell)
        reasons = _layout_escapability_reasons(
            map_model, attempt_rng, stage=stage, spawn_native=spawn_native
        )
        if not reasons:
            metadata = {
                **metadata,
                "escapability_validated": True,
                "escapability_stage_budget_ticks": _STAGE_ESCAPE_TICKS[stage],
                "escapability_attempts": attempt + 1,
            }
            return map_model, spawn_cell, metadata, attempt_rng
        last_reasons = reasons
    raise ValueError(
        f"Could not generate an escapable {stage!r} layout for template {template!r} "
        f"after {_MAX_LAYOUT_ATTEMPTS} attempts: {'; '.join(last_reasons)}"
    )


def generate_synthetic_curriculum(
    output_directory: str | Path,
    *,
    count: int = 12,
    seed: int = 20260804,
    reference_model: RecordedWorldModel | None = None,
    overwrite: bool = False,
) -> Path:
    if count < 3:
        raise ValueError("Synthetic curriculum requires at least three variants")
    return generate_curriculum_from_plan(
        output_directory,
        _variant_plan(count),
        seed=seed,
        reference_model=reference_model,
        overwrite=overwrite,
    )


def generate_curriculum_from_plan(
    output_directory: str | Path,
    plan: list[tuple[str, str, str, str, int]],
    *,
    seed: int,
    reference_model: RecordedWorldModel | None = None,
    overwrite: bool = False,
    curriculum_name: str = "FlyFF generic open-farm curriculum",
) -> Path:
    """Generate a curriculum from an explicit (stage, template, density,
    respawn, obstacle_level) plan, bypassing _variant_plan's automatic
    cycling.

    _variant_plan's stage cycle (period 3) and template cycle (period 6)
    share a common factor, so every stage it produces is structurally
    limited to exactly 2 of the 6 templates no matter how many variants are
    requested -- fine for a training curriculum that has already been
    trained on and must stay reproducible, but not adequate for a held-out
    set meant to probe generalization across the full template space. Pass
    an explicit plan here instead of going through generate_synthetic_curriculum
    when that coverage matters.
    """

    if not plan:
        raise ValueError("Synthetic curriculum requires at least one planned variant")
    root = Path(output_directory)
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    variants: list[SyntheticVariantSpec] = []
    reference_movement = _movement_from_reference(reference_model)
    for index, (stage, template, density, respawn, obstacle_level) in enumerate(plan, start=1):
        variant_seed = int(seed + index * 7919)
        map_model, spawn_cell, map_metadata, rng = _generate_validated_layout(
            template,
            variant_seed,
            obstacle_level=obstacle_level,
            stage=stage,
        )
        name = f"{index:02d}_{stage}_{template}_{density}_{respawn}"
        variant_root = root / "variants" / name
        map_root = variant_root / "map_assets"
        map_model.save_assets(
            map_root,
            metadata={
                **map_metadata,
                "variant": name,
                "stage": stage,
                "density_profile": density,
                "respawn_profile": respawn,
                "seed": variant_seed,
            },
        )
        population, cluster_count, _density_scale = _density_configuration(density, rng)
        section_count = int(rng.integers(4, 7))
        reservoirs = _clustered_spawn_reservoirs(
            map_model,
            rng,
            section_count=section_count,
            cluster_count=cluster_count,
            density_profile=density,
        )
        probabilities = _section_probabilities(reservoirs, rng, density_profile=density)
        transition = _transition_matrix(probabilities, density_profile=density)
        spawn_native = map_model.layout_to_native(*spawn_cell)
        world = RecordedWorldModel(
            schema_version=5,
            source_recordings=(f"synthetic:{name}",),
            section_count=section_count,
            hub_section=section_count,
            population_median=population,
            section_population_probabilities=probabilities,
            player_start_positions=(spawn_native,),
            spawn_positions_by_section=reservoirs,
            transition_probabilities=transition,
            respawn_delay_seconds=_respawn_samples(respawn, rng),
            movement=reference_movement,
            monster_speed_cells_per_second=float(rng.uniform(0.0, 0.18)),
            frame_interval_seconds=(reference_model.frame_interval_seconds if reference_model else 0.20),
            native_units_per_cell=map_model.native_units_per_cell,
            recording_frame_interval_seconds=(reference_model.recording_frame_interval_seconds if reference_model else 0.20),
            cast_step_seconds=(reference_model.cast_step_seconds if reference_model else 0.80),
            cast_movement_seconds=(reference_model.cast_movement_seconds if reference_model else 0.20),
            respawn_model_mode=("synthetic_redistribution_heavy" if density == "shifting" else "synthetic_global_redistribution"),
            respawn_delay_source=f"synthetic_{respawn}",
            human_action_probabilities=(0.60, 0.14, 0.14, 0.10, 0.02),
            # This layout's escapability was validated against the live
            # calibrated-arc kernel (see _regains_movement_within), so the
            # world is genuinely a calibrated-arc-physics curriculum entry
            # -- even though `movement` above still carries the legacy
            # per-action stats (kept only as informational provenance, see
            # MovementModel's docstring; RecordedFarmingEnv never reads it).
            movement_physics_model=MOVEMENT_PHYSICS_MODEL_ID,
            fit_warnings=(
                "Synthetic environment: use for generic farming pretraining, not for Tower-specific validation.",
                "Layout intentionally favors large open farming regions and excludes dungeon/corridor generation.",
            ),
        )
        world_path = variant_root / "world.json.gz"
        world.save(world_path)
        variants.append(
            SyntheticVariantSpec(
                name=name,
                stage=stage,
                template=template,
                density_profile=density,
                respawn_profile=respawn,
                map_assets=str(map_root.relative_to(root)).replace("\\", "/"),
                world_model=str(world_path.relative_to(root)).replace("\\", "/"),
                weight=1.0,
                seed=variant_seed,
                notes="Large open farming layout; no procedural corridors or dungeon rooms.",
            )
        )
    curriculum = SyntheticCurriculum(
        schema_version=CURRICULUM_SCHEMA_VERSION,
        name=curriculum_name,
        generated_seed=int(seed),
        variants=tuple(variants),
        design_rules=(
            "Episodes start at one designated spawn point per map.",
            "Maps are large and mostly open.",
            "Walls are partial and openings remain broad.",
            "Scattered obstacles are sparse.",
            "No maze, dungeon-room chain, long corridor, or precision-navigation layout is generated.",
            "Monster populations use loose spatial clusters and varied regional density.",
            "Respawn timing and redistribution vary across layouts.",
            "Every layout is validated so the bot's real latched-forward controls can "
            "regain meaningful movement from any sampled obstacle-adjacent position and "
            "heading within a stage-appropriate tick budget; unescapable pockets are "
            "rejected and regenerated, never shipped.",
        ),
    )
    return curriculum.save(root / "curriculum.json")


class SyntheticCurriculumEnv(_BaseEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        curriculum_path: str | Path,
        *,
        stage: str = "all",
        seed: int = 0,
        episode_steps: int = 6_000,
        episode_seconds: float | None = None,
    ) -> None:
        self.manifest_path = Path(curriculum_path).resolve()
        self.curriculum = SyntheticCurriculum.load(self.manifest_path)
        self.entries = self.curriculum.select(stage)
        self.rng = np.random.default_rng(seed)
        self.episode_steps = int(episode_steps)
        self.episode_seconds = (
            None if episode_seconds is None else float(episode_seconds)
        )
        weights = np.asarray([max(0.0, entry.weight) for entry in self.entries], dtype=np.float64)
        self.weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1.0 / len(weights))
        self._cache: dict[str, RecordedFarmingEnv] = {}
        self._current: RecordedFarmingEnv | None = None
        self.current_variant = ""
        first = self._environment_for(self.entries[0], seed=seed)
        self.action_space = first.action_space
        self.observation_space = first.observation_space

    def _environment_for(self, entry: SyntheticVariantSpec, *, seed: int) -> RecordedFarmingEnv:
        cached = self._cache.get(entry.name)
        if cached is not None:
            return cached
        root = self.manifest_path.parent
        world = RecordedWorldModel.load(root / entry.world_model)
        map_model = MapModel.load(root / entry.map_assets)
        env = RecordedFarmingEnv(
            world,
            map_model=map_model,
            seed=seed,
            episode_steps=self.episode_steps,
            episode_seconds=self.episode_seconds,
        )
        self._cache[entry.name] = env
        return env

    @property
    def map(self) -> MapModel:
        if self._current is None:
            return next(iter(self._cache.values())).map
        return self._current.map

    @property
    def model(self) -> RecordedWorldModel:
        if self._current is None:
            return next(iter(self._cache.values())).model
        return self._current.model

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        index = int(self.rng.choice(np.arange(len(self.entries)), p=self.weights))
        entry = self.entries[index]
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        self._current = self._environment_for(entry, seed=episode_seed)
        self.current_variant = entry.name
        observation, info = self._current.reset(seed=episode_seed, options=options)
        info = dict(info)
        info["synthetic_variant"] = entry.name
        info["synthetic_stage"] = entry.stage
        return observation, info

    def step(self, action: int):
        if self._current is None:
            raise RuntimeError("reset() must be called before step()")
        observation, reward, terminated, truncated, info = self._current.step(action)
        info = dict(info)
        info["synthetic_variant"] = self.current_variant
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        for env in self._cache.values():
            close = getattr(env, "close", None)
            if callable(close):
                close()


def curriculum_summary(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    curriculum = SyntheticCurriculum.load(manifest_path)
    stage_counts: dict[str, int] = {}
    templates: dict[str, int] = {}
    densities: dict[str, int] = {}
    respawns: dict[str, int] = {}
    populations: list[int] = []
    for item in curriculum.variants:
        stage_counts[item.stage] = stage_counts.get(item.stage, 0) + 1
        templates[item.template] = templates.get(item.template, 0) + 1
        densities[item.density_profile] = densities.get(item.density_profile, 0) + 1
        respawns[item.respawn_profile] = respawns.get(item.respawn_profile, 0) + 1
        world = RecordedWorldModel.load(manifest_path.parent / item.world_model)
        populations.append(world.population_median)
    return {
        "schema_version": curriculum.schema_version,
        "name": curriculum.name,
        "variants": len(curriculum.variants),
        "stages": stage_counts,
        "templates": templates,
        "density_profiles": densities,
        "respawn_profiles": respawns,
        "population_range": [min(populations), max(populations)] if populations else [0, 0],
        "design_rules": list(curriculum.design_rules),
    }


def iter_variant_environments(
    curriculum_path: str | Path,
    *,
    stage: str = "all",
    seed: int = 0,
    episode_steps: int = 6_000,
    episode_seconds: float | None = None,
    variant_name: str | None = None,
) -> Iterable[tuple[SyntheticVariantSpec, RecordedFarmingEnv]]:
    manifest_path = Path(curriculum_path).resolve()
    curriculum = SyntheticCurriculum.load(manifest_path)
    root = manifest_path.parent
    selected = curriculum.select(stage)
    if variant_name is not None:
        selected = tuple(item for item in selected if item.name == variant_name)
        if not selected:
            raise ValueError(
                f"Synthetic variant {variant_name!r} is not available in stage {stage!r}"
            )
    for index, entry in enumerate(selected):
        yield entry, RecordedFarmingEnv(
            RecordedWorldModel.load(root / entry.world_model),
            map_model=MapModel.load(root / entry.map_assets),
            seed=seed + index * 1009,
            episode_steps=episode_steps,
            episode_seconds=episode_seconds,
        )

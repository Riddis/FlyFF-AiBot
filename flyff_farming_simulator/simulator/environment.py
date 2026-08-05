from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from farming.actions import (
    FarmingAction,
    FarmingCommand,
    FarmingEvent,
    POLICY_ACTION_NVECS,
    SteeringAction,
    coerce_farming_command,
)
from farming.map_features import DirectPathState, MapCellRisk
from farming.observation import (
    ActorObservation,
    ObservationBuilder,
    ObservationFrame,
    ObservationScales,
    PlayerObservation,
)
from farming.session import SessionOutcome

from .map_model import MapModel
from .reward_model import (
    SimulatorRewardCalculator,
    SimulatorRewardComponents,
    SimulatorRewardConfig,
    SimulatorRewardEvidence,
)
from .world_model import RecordedWorldModel

try:  # Optional at import time; required only for PPO training.
    import gymnasium as gym
    from gymnasium import spaces

    _BaseEnv = gym.Env
except ImportError:  # pragma: no cover - exercised on minimal installations.
    gym = None
    spaces = None

    class _BaseEnv:  # type: ignore[no-redef]
        pass


@dataclass(slots=True)
class SimActor:
    actor_id: int
    x: float
    z: float
    alive: bool = True
    respawn_at: float = math.inf
    death_section: int = 0
    wander_heading: float = 0.0


class RecordedFarmingEnv(_BaseEnv):
    """Interactive unified-policy simulator fitted from human recordings.

    The observation vector is built by the copied production
    ``native-unified-923-v4`` ObservationBuilder. The environment therefore has
    the same 923 float inputs as the live policy. The policy action is now
    MultiDiscrete([3, 3]): steering (straight/left/right) plus an independent
    event (none/EVA/jump). Forward is latched while control is active.

    EVA and jump no longer compete with steering. A successful EVA cast consumes
    its cast duration while movement is allowed for only
    ``cast_movement_seconds`` to approximate the animation lock.
    """

    metadata = {"render_modes": []}
    _DIRECT_PATH_LIMIT = 96

    def __init__(
        self,
        model: RecordedWorldModel,
        *,
        map_model: MapModel | None = None,
        seed: int | None = None,
        episode_steps: int = 6_000,
        episode_seconds: float | None = None,
        vision_radius_cells: float = 50.0,
        eva_radius_cells: float = 8.0,
        eva_cooldown_seconds: float = 2.0,
        jump_cooldown_seconds: float = 2.0,
        reward_config: SimulatorRewardConfig | None = None,
    ) -> None:
        self.model = model
        self.map = map_model or MapModel.load()
        self.rng = np.random.default_rng(seed)
        self.episode_steps = int(episode_steps)
        if self.episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if episode_seconds is not None and (
            not math.isfinite(float(episode_seconds)) or float(episode_seconds) <= 0.0
        ):
            raise ValueError("episode_seconds must be finite and positive when provided")
        self.episode_seconds = (
            None if episode_seconds is None else float(episode_seconds)
        )
        self.vision_radius_cells = float(vision_radius_cells)
        self.eva_radius_cells = float(eva_radius_cells)
        self.eva_cooldown_seconds = float(eva_cooldown_seconds)
        self.jump_cooldown_seconds = float(jump_cooldown_seconds)
        self.movement_dt = float(max(0.02, model.frame_interval_seconds))
        self.cast_dt = float(max(self.movement_dt, model.cast_step_seconds))
        self.cast_movement_seconds = float(
            np.clip(model.cast_movement_seconds, 0.0, self.cast_dt)
        )
        self.observation_builder = ObservationBuilder(
            ObservationScales(
                vision_radius_cells=self.vision_radius_cells,
                eva_radius_cells=self.eva_radius_cells,
            )
        )
        self.reward_calculator = SimulatorRewardCalculator(
            reward_config or SimulatorRewardConfig()
        )
        if spaces is not None:
            self.action_space = spaces.MultiDiscrete(np.asarray(POLICY_ACTION_NVECS, dtype=np.int64))
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.observation_builder.observation_size,),
                dtype=np.float32,
            )
        self.actors: list[SimActor] = []
        self.player_x = 0.0
        self.player_z = 0.0
        self.heading = 0.0
        self.elapsed = 0.0
        self.steps = 0
        self.last_eva_at = -math.inf
        self.last_jump_at = -math.inf
        self.last_action = FarmingAction.RUN_FORWARD
        self.held_movement: FarmingAction | None = FarmingAction.RUN_FORWARD
        self.last_displacement_cells = 0.0
        self.last_contact = False
        self.total_kills = 0
        self.total_distance_cells = 0.0
        self.start_x = 0.0
        self.start_z = 0.0
        self.contact_count = 0
        self.total_eva_attempts = 0
        self.total_valid_eva_casts = 0
        self.total_invalid_eva_attempts = 0
        self.total_missed_eva_opportunities = 0
        self.reward_component_totals = SimulatorRewardComponents().as_dict()
        self._approach_potential_cells = 0.0
        self._best_group_actor_id: int | None = None
        self._nearest_reachable_actor_id: int | None = None
        self._visited_cells: set[tuple[int, int]] = set()
        self._next_actor_id = 1
        self._spawn_positions = tuple(
            tuple(dict.fromkeys((float(x), float(z)) for x, z in section))
            for section in self.model.spawn_positions_by_section
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.elapsed = 0.0
        self.steps = 0
        self.last_eva_at = -math.inf
        self.last_jump_at = -math.inf
        self.last_action = FarmingAction.RUN_FORWARD
        self.held_movement = FarmingAction.RUN_FORWARD
        self.last_displacement_cells = 0.0
        self.last_contact = False
        self.total_kills = 0
        self.total_distance_cells = 0.0
        self.contact_count = 0
        self.total_eva_attempts = 0
        self.total_valid_eva_casts = 0
        self.total_invalid_eva_attempts = 0
        self.total_missed_eva_opportunities = 0
        self.reward_component_totals = SimulatorRewardComponents().as_dict()
        self._approach_potential_cells = 0.0
        self._best_group_actor_id = None
        self._nearest_reachable_actor_id = None
        self._visited_cells = set()
        start = self.model.player_start_positions[
            int(self.rng.integers(0, len(self.model.player_start_positions)))
        ]
        self.player_x, self.player_z = float(start[0]), float(start[1])
        if self.map.features.cell_risk(self.map.native_to_layout_cell(self.player_x, self.player_z)) not in {
            MapCellRisk.SAFE,
            MapCellRisk.OBSTACLE_BUFFER,
        }:
            self.player_x, self.player_z = self.map.layout_to_native(*self.map.random_safe_cell(self.rng))
        self.start_x, self.start_z = self.player_x, self.player_z
        cell = self.map.native_to_layout_cell(self.player_x, self.player_z)
        if cell is not None:
            self._visited_cells.add(cell)
        self.heading = float(self.rng.uniform(-math.pi, math.pi))
        self.actors = []
        self._next_actor_id = 1
        section_probabilities = np.asarray(
            self.model.section_population_probabilities, dtype=np.float64
        )
        section_total = self.model.section_count + 1
        if (
            section_probabilities.shape != (section_total,)
            or not np.all(np.isfinite(section_probabilities))
            or float(section_probabilities.sum()) <= 0.0
        ):
            section_probabilities = np.full(section_total, 1.0 / section_total)
        else:
            section_probabilities = section_probabilities / section_probabilities.sum()
        occupied_positions: set[tuple[float, float]] = set()
        for _ in range(self.model.population_median):
            section = int(self.rng.choice(np.arange(section_total), p=section_probabilities))
            x, z = self._sample_spawn_position(section, occupied=occupied_positions)
            occupied_positions.add((x, z))
            self.actors.append(
                SimActor(
                    actor_id=self._next_actor_id,
                    x=x,
                    z=z,
                    wander_heading=float(self.rng.uniform(-math.pi, math.pi)),
                )
            )
            self._next_actor_id += 1
        observation = self._observation()
        info = self._info(kills=0, reward_components={})
        return observation, info

    def step(self, action: object):
        command = coerce_farming_command(
            action,
            legacy_event_steering={
                FarmingAction.RUN_FORWARD: SteeringAction.STRAIGHT,
                FarmingAction.RUN_FORWARD_LEFT: SteeringAction.LEFT,
                FarmingAction.RUN_FORWARD_RIGHT: SteeringAction.RIGHT,
            }.get(self.held_movement, SteeringAction.STRAIGHT),
        )
        movement_action = command.movement_action
        selected = command.legacy_action
        before_approach_potential = float(self._approach_potential_cells)
        kills = 0
        contact = False
        jump = False
        displacement = 0.0
        eva_available = self.elapsed - self.last_eva_at >= self.eva_cooldown_seconds
        eva_targets_before_action = self._eva_count()
        missed_eva_opportunity = bool(
            not command.eva_requested
            and eva_available
            and eva_targets_before_action
            >= self.reward_calculator.config.missed_eva_minimum_targets
        )
        self.total_missed_eva_opportunities += int(missed_eva_opportunity)
        nominal_elapsed_step = (
            self.cast_dt if command.eva_requested and eva_available else self.movement_dt
        )
        elapsed_step = nominal_elapsed_step
        if self.episode_seconds is not None:
            remaining = max(0.0, float(self.episode_seconds) - self.elapsed)
            elapsed_step = min(elapsed_step, remaining)
        movement_time = min(self.movement_dt, elapsed_step)

        # W/Z is implicit and always held. Steering is applied before the event
        # so left/right movement continues during EVA and jump.
        self.held_movement = movement_action
        if command.eva_requested:
            self.total_eva_attempts += 1
            self.last_eva_at = self.elapsed
            if eva_available:
                self.total_valid_eva_casts += 1
                movement_time = min(self.cast_movement_seconds, elapsed_step)
                if movement_time > 0.0:
                    displacement, contact = self._move_player(
                        movement_action,
                        distance_scale=movement_time / self.movement_dt,
                    )
                kills = self._cast_eva()
            else:
                self.total_invalid_eva_attempts += 1
                if movement_time > 0.0:
                    displacement, contact = self._move_player(
                        movement_action,
                        distance_scale=movement_time / self.movement_dt,
                    )
        else:
            jump = command.jump_requested and (
                self.elapsed - self.last_jump_at >= self.jump_cooldown_seconds
            )
            if jump:
                self.last_jump_at = self.elapsed
            if movement_time > 0.0:
                displacement, contact = self._move_player(
                    movement_action,
                    distance_scale=movement_time / self.movement_dt,
                )

        self.last_displacement_cells = displacement
        self.total_distance_cells += displacement
        self.contact_count += int(contact)

        movement_candidates = self._visible_candidates()
        player_cell = self.map.native_to_layout_cell(self.player_x, self.player_z)
        geodesic_field = self._geodesic_field(player_cell)
        movement_potential, _target = self._group_approach_potential(
            movement_candidates, geodesic_field
        )
        approach_progress_cells = (
            movement_potential - before_approach_potential
            if not command.eva_requested
            else 0.0
        )

        world_changed = self._move_monsters(elapsed_step)
        self.elapsed += elapsed_step
        self.steps += 1
        world_changed = self._respawn_due_actors() or world_changed
        if player_cell is not None:
            self._visited_cells.add(player_cell)
        risk = self.map.features.cell_risk(player_cell)
        forbidden_distance = self.map.features.forbidden_distance(player_cell)

        self.last_action = selected
        self.last_contact = contact
        observation_candidates = (
            self._visible_candidates() if world_changed else movement_candidates
        )
        observation = self._observation(
            geodesic_field=geodesic_field,
            candidates=observation_candidates,
        )
        reward_result = self.reward_calculator.calculate(
            SimulatorRewardEvidence(
                native_kill_delta=kills,
                approach_progress_cells=approach_progress_cells,
                eva_attempted=command.eva_requested,
                eva_available=eva_available,
                eva_target_count_before_action=eva_targets_before_action,
                contact=contact,
                map_cell_risk=risk,
                jump_performed=jump,
                forbidden_distance_cells=forbidden_distance,
                session_outcome=SessionOutcome.continuing(),
            )
        )
        components = reward_result.components.as_dict()
        for name, value in components.items():
            self.reward_component_totals[name] = (
                self.reward_component_totals.get(name, 0.0) + float(value)
            )

        self.total_kills += kills
        terminated = risk is MapCellRisk.TELEPORT_TRIGGER
        time_limit_reached = bool(
            self.episode_seconds is not None
            and self.elapsed + 1.0e-9 >= self.episode_seconds
        )
        truncated = time_limit_reached or self.steps >= self.episode_steps
        info = self._info(kills=kills, reward_components=components)
        info.update(
            {
                "steering": command.steering.name,
                "steering_index": int(command.steering),
                "event": command.event.name,
                "event_index": int(command.event),
                "factorized_action": list(command.as_array()),
            }
        )
        return observation, float(reward_result.total), terminated, truncated, info

    def _sample_spawn_position(
        self,
        section: int,
        *,
        occupied: set[tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        sections = self._spawn_positions
        selected = sections[section] if 0 <= section < len(sections) else ()
        candidates = selected or tuple(position for group in sections for position in group)
        for _ in range(100):
            x, z = candidates[int(self.rng.integers(0, len(candidates)))]
            position = (float(x), float(z))
            if occupied is not None and position in occupied:
                continue
            cell = self.map.native_to_layout_cell(x, z)
            if self.map.features.cell_risk(cell) in {
                MapCellRisk.SAFE,
                MapCellRisk.OBSTACLE_BUFFER,
            }:
                return position
        for _ in range(200):
            position = self.map.layout_to_native(*self.map.random_safe_cell(self.rng))
            if occupied is None or position not in occupied:
                return position
        raise RuntimeError("Could not place an actor at a distinct safe spawn position")

    def _cast_eva(self) -> int:
        killed = 0
        for actor in self.actors:
            if not actor.alive:
                continue
            distance_cells = math.hypot(actor.x - self.player_x, actor.z - self.player_z)
            distance_cells /= self.map.native_units_per_cell
            if distance_cells <= self.eva_radius_cells:
                actor.alive = False
                actor.death_section = self.map.section(
                    actor.x,
                    actor.z,
                    section_count=self.model.section_count,
                )
                delay = self.model.respawn_delay_seconds[
                    int(self.rng.integers(0, len(self.model.respawn_delay_seconds)))
                ]
                actor.respawn_at = self.elapsed + max(self.movement_dt, float(delay))
                killed += 1
        return killed

    def _respawn_due_actors(self) -> bool:
        transition = np.asarray(self.model.transition_probabilities, dtype=np.float64)
        occupied = {(actor.x, actor.z) for actor in self.actors if actor.alive}
        changed = False
        for actor in self.actors:
            if actor.alive or actor.respawn_at > self.elapsed:
                continue
            row = transition[min(actor.death_section, transition.shape[0] - 1)]
            target = int(self.rng.choice(np.arange(len(row)), p=row / row.sum()))
            actor.x, actor.z = self._sample_spawn_position(target, occupied=occupied)
            occupied.add((actor.x, actor.z))
            actor.alive = True
            actor.respawn_at = math.inf
            actor.wander_heading = float(self.rng.uniform(-math.pi, math.pi))
            changed = True
        return changed

    def _move_player(
        self,
        action: FarmingAction,
        *,
        distance_scale: float = 1.0,
    ) -> tuple[float, bool]:
        model = self.model.movement[int(action)]
        distance = float(
            np.clip(
                self.rng.normal(model.distance_mean_cells, max(0.01, model.distance_std_cells)),
                0.0,
                5.0,
            )
        ) * max(0.0, float(distance_scale))
        turn = float(
            self.rng.normal(model.turn_mean_radians, max(0.005, model.turn_std_radians))
        ) * max(0.0, float(distance_scale))
        if action is FarmingAction.RUN_FORWARD_LEFT:
            turn = abs(turn) if abs(turn) > 0.01 else 0.10
        elif action is FarmingAction.RUN_FORWARD_RIGHT:
            turn = -abs(turn) if abs(turn) > 0.01 else -0.10
        else:
            turn = 0.0 if action is not FarmingAction.RUN_FORWARD_JUMP else turn * 0.15
        self.heading = math.atan2(math.sin(self.heading + turn), math.cos(self.heading + turn))
        native_distance = distance * self.map.native_units_per_cell
        target_x = self.player_x + math.cos(self.heading) * native_distance
        target_z = self.player_z + math.sin(self.heading) * native_distance
        accepted_x, accepted_z = self.player_x, self.player_z
        contact = False
        samples = max(2, int(math.ceil(distance * 4.0)))
        for index in range(1, samples + 1):
            fraction = index / samples
            x = self.player_x + (target_x - self.player_x) * fraction
            z = self.player_z + (target_z - self.player_z) * fraction
            risk = self.map.features.cell_risk(self.map.native_to_layout_cell(x, z))
            if risk in {MapCellRisk.OBSTACLE, MapCellRisk.OUTSIDE_OR_UNKNOWN}:
                contact = True
                break
            accepted_x, accepted_z = x, z
            if risk is MapCellRisk.TELEPORT_TRIGGER:
                break
        actual = math.hypot(accepted_x - self.player_x, accepted_z - self.player_z)
        actual /= self.map.native_units_per_cell
        self.player_x, self.player_z = accepted_x, accepted_z
        return float(actual), contact

    def _move_monsters(self, elapsed_seconds: float) -> bool:
        speed = max(0.0, self.model.monster_speed_cells_per_second)
        if speed <= 0.0:
            return False
        native_distance = speed * elapsed_seconds * self.map.native_units_per_cell
        changed = False
        for actor in self.actors:
            if not actor.alive:
                continue
            if self.rng.random() < 0.06:
                actor.wander_heading += float(self.rng.normal(0.0, 0.8))
            target_x = actor.x + math.cos(actor.wander_heading) * native_distance
            target_z = actor.z + math.sin(actor.wander_heading) * native_distance
            risk = self.map.features.cell_risk(self.map.native_to_layout_cell(target_x, target_z))
            if risk in {MapCellRisk.SAFE, MapCellRisk.OBSTACLE_BUFFER}:
                actor.x, actor.z = target_x, target_z
                changed = True
            else:
                actor.wander_heading += math.pi * float(self.rng.uniform(0.5, 1.5))
        return changed

    def _eva_count(self) -> int:
        return sum(
            actor.alive
            and math.hypot(actor.x - self.player_x, actor.z - self.player_z)
            / self.map.native_units_per_cell
            <= self.eva_radius_cells
            for actor in self.actors
        )

    def nearest_actor_relative_angle(self) -> float | None:
        nearest: tuple[float, SimActor] | None = None
        for actor in self.actors:
            if not actor.alive:
                continue
            dx = actor.x - self.player_x
            dz = actor.z - self.player_z
            distance = math.hypot(dx, dz)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, actor)
        if nearest is None:
            return None
        return self._relative_angle_to_actor(nearest[1])

    def nearest_reachable_relative_angle(self) -> float | None:
        """Angle to the nearest actor proven reachable in the last observation."""

        if self._nearest_reachable_actor_id is None:
            return None
        for actor in self.actors:
            if actor.alive and actor.actor_id == self._nearest_reachable_actor_id:
                return self._relative_angle_to_actor(actor)
        return None

    def movement_path_clear(self, action: FarmingAction, *, lookahead_cells: float = 4.0) -> bool:
        """Conservatively check the nominal segment used by a scripted policy."""

        selected = FarmingAction(int(action))
        model = self.model.movement[int(selected)]
        turn = float(model.turn_mean_radians)
        if selected is FarmingAction.RUN_FORWARD_LEFT:
            turn = abs(turn) if abs(turn) > 0.01 else 0.10
        elif selected is FarmingAction.RUN_FORWARD_RIGHT:
            turn = -abs(turn) if abs(turn) > 0.01 else -0.10
        else:
            turn = 0.0
        heading = self.heading + turn
        distance = min(max(0.5, float(model.distance_mean_cells)), float(lookahead_cells))
        for sample in range(1, max(2, int(math.ceil(distance * 4.0))) + 1):
            along = distance * sample / max(2, int(math.ceil(distance * 4.0)))
            cell = self.map.native_to_layout_cell(
                self.player_x + math.cos(heading) * along * self.map.native_units_per_cell,
                self.player_z + math.sin(heading) * along * self.map.native_units_per_cell,
            )
            if self.map.features.cell_risk(cell) in {
                MapCellRisk.OBSTACLE,
                MapCellRisk.OUTSIDE_OR_UNKNOWN,
                MapCellRisk.TELEPORT_TRIGGER,
            }:
                return False
        return True

    def best_group_relative_angle(self) -> float | None:
        """Angle to the current reachable group target used by the reward audit."""

        if self._best_group_actor_id is None:
            return self.nearest_actor_relative_angle()
        for actor in self.actors:
            if actor.alive and actor.actor_id == self._best_group_actor_id:
                return self._relative_angle_to_actor(actor)
        return self.nearest_actor_relative_angle()

    def _relative_angle_to_actor(self, actor: SimActor) -> float:
        target = math.atan2(actor.z - self.player_z, actor.x - self.player_x)
        return math.atan2(math.sin(target - self.heading), math.cos(target - self.heading))

    def eva_available(self) -> bool:
        return self.elapsed - self.last_eva_at >= self.eva_cooldown_seconds

    def eva_target_count(self) -> int:
        return self._eva_count()

    def _visible_candidates(self) -> list[tuple[float, SimActor, float, float]]:
        candidates: list[tuple[float, SimActor, float, float]] = []
        for actor in self.actors:
            if not actor.alive:
                continue
            dx_cells = (actor.x - self.player_x) / self.map.native_units_per_cell
            dz_cells = (actor.z - self.player_z) / self.map.native_units_per_cell
            direct_distance = math.hypot(dx_cells, dz_cells)
            if direct_distance <= self.vision_radius_cells:
                candidates.append((direct_distance, actor, dx_cells, dz_cells))
        candidates.sort(key=lambda item: (item[0], item[1].actor_id))
        return candidates

    def _geodesic_field(
        self,
        player_cell: tuple[int, int] | None,
    ) -> dict[tuple[int, int], float]:
        if player_cell is None:
            return {}
        return self.map.features.bounded_geodesic_field(
            player_cell,
            maximum_distance_cells=self.vision_radius_cells * 1.5,
        )

    def _group_approach_potential(
        self,
        candidates: list[tuple[float, SimActor, float, float]],
        geodesic_field: dict[tuple[int, int], float],
    ) -> tuple[float, int | None]:
        """Return a reachable-group potential measured in effective cells.

        Each visible actor represents a possible group center. Nearby actors add
        a capped density bonus, while geodesic distance subtracts from utility.
        The step reward uses only the change caused by player movement, so actor
        wandering and respawns cannot manufacture approach reward.
        """

        no_target_potential = -self.vision_radius_cells * 1.5
        if not candidates or not geodesic_field:
            return no_target_potential, None
        radius = max(1.0e-6, self.eva_radius_cells)
        inverse = 1.0 / radius
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, (_distance, _actor, dx_cells, dz_cells) in enumerate(candidates):
            key = (math.floor(dx_cells * inverse), math.floor(dz_cells * inverse))
            buckets.setdefault(key, []).append(index)

        radius_squared = radius * radius
        best_score = -math.inf
        best_actor_id: int | None = None
        # The nearest 96 visible actors are enough to identify an actionable
        # group target while keeping dense-map reward audits responsive. All
        # visible actors still contribute to each candidate's local group count.
        for index, (_distance, actor, dx_cells, dz_cells) in enumerate(
            candidates[: self._DIRECT_PATH_LIMIT]
        ):
            key_x = math.floor(dx_cells * inverse)
            key_z = math.floor(dz_cells * inverse)
            local_count = 0
            for bucket_x in range(key_x - 1, key_x + 2):
                for bucket_z in range(key_z - 1, key_z + 2):
                    for other_index in buckets.get((bucket_x, bucket_z), ()):
                        other = candidates[other_index]
                        delta_x = other[2] - dx_cells
                        delta_z = other[3] - dz_cells
                        if delta_x * delta_x + delta_z * delta_z <= radius_squared:
                            local_count += 1
                            if local_count >= 16:
                                break
                    if local_count >= 16:
                        break
                if local_count >= 16:
                    break
            actor_cell = self.map.native_to_layout_cell(actor.x, actor.z)
            if actor_cell is None:
                continue
            geodesic = float(geodesic_field.get(actor_cell, math.inf))
            if not math.isfinite(geodesic):
                continue
            density_bonus_cells = 0.75 * min(max(0, local_count - 1), 12)
            score = density_bonus_cells - geodesic
            if score > best_score:
                best_score = score
                best_actor_id = actor.actor_id
        if not math.isfinite(best_score):
            return no_target_potential, None
        return float(best_score), best_actor_id

    def _observation(
        self,
        *,
        geodesic_field: dict[tuple[int, int], float] | None = None,
        candidates: list[tuple[float, SimActor, float, float]] | None = None,
    ) -> np.ndarray:
        player_cell = self.map.native_to_layout_cell(self.player_x, self.player_z)
        player_layout = self.map.native_to_layout_cells(self.player_x, self.player_z)
        if candidates is None:
            candidates = self._visible_candidates()
        direct_ids = {item[1].actor_id for item in candidates[: self._DIRECT_PATH_LIMIT]}
        if geodesic_field is None:
            geodesic_field = self._geodesic_field(player_cell)
        potential, best_actor_id = self._group_approach_potential(
            candidates, geodesic_field
        )
        self._approach_potential_cells = potential
        self._best_group_actor_id = best_actor_id
        nearest_reachable: tuple[float, int] | None = None

        actors: list[ActorObservation] = []
        for _distance, actor, dx_cells, dz_cells in candidates:
            actor_layout = self.map.native_to_layout_cells(actor.x, actor.z)
            actor_cell = self.map.native_to_layout_cell(actor.x, actor.z)
            geodesic = (
                math.inf
                if actor_cell is None
                else geodesic_field.get(actor_cell, math.inf)
            )
            if math.isfinite(geodesic) and (
                nearest_reachable is None or geodesic < nearest_reachable[0]
            ):
                nearest_reachable = (float(geodesic), actor.actor_id)
            direct_path = DirectPathState.UNKNOWN
            if actor.actor_id in direct_ids:
                direct_path = self.map.features.direct_path_state(player_cell, actor_cell)
            actors.append(
                ActorObservation(
                    actor_id=actor.actor_id,
                    legacy_dx_cells=actor_layout[0] - player_layout[0],
                    legacy_dy_cells=actor_layout[1] - player_layout[1],
                    direct_dx_cells=dx_cells,
                    direct_dz_cells=dz_cells,
                    geodesic_cells=geodesic,
                    direct_path=direct_path,
                    alive=True,
                )
            )
        self._nearest_reachable_actor_id = (
            None if nearest_reachable is None else nearest_reachable[1]
        )
        normalized_x, normalized_z = self.map.features.normalized_position(player_cell)
        eva_cooldown = float(
            np.clip((self.elapsed - self.last_eva_at) / self.eva_cooldown_seconds, 0.0, 1.0)
        )
        jump_cooldown = float(
            np.clip((self.elapsed - self.last_jump_at) / self.jump_cooldown_seconds, 0.0, 1.0)
        )
        built = self.observation_builder.build(
            ObservationFrame(
                player=PlayerObservation(
                    normalized_x=normalized_x,
                    normalized_z=normalized_z,
                    heading_radians=self.heading,
                    eva_cooldown_fraction=eva_cooldown,
                    displacement_cells=self.last_displacement_cells,
                    contact=self.last_contact,
                    held_movement=self.held_movement,
                    last_policy_action=self.last_action,
                    jump_cooldown_fraction=jump_cooldown,
                    map_available=True,
                ),
                actors=tuple(actors),
                local_map=self.map.features.local_crop(player_cell, side=11),
                context_map=self.map.features.context_crop(player_cell),
            )
        )
        return built.vector

    def close(self) -> None:
        return None

    def _info(self, *, kills: int, reward_components: dict[str, float]) -> dict[str, Any]:
        net_displacement = math.hypot(self.player_x - self.start_x, self.player_z - self.start_z)
        net_displacement /= self.map.native_units_per_cell
        return {
            "kills": int(kills),
            "total_kills": int(self.total_kills),
            "living_monsters": sum(actor.alive for actor in self.actors),
            "eva_count": self._eva_count(),
            "player_x": self.player_x,
            "player_z": self.player_z,
            "elapsed_seconds": float(self.elapsed),
            "total_distance_cells": float(self.total_distance_cells),
            "net_displacement_cells": float(net_displacement),
            "path_efficiency": float(net_displacement / max(1.0e-9, self.total_distance_cells)),
            "unique_cells": len(self._visited_cells),
            "contacts": int(self.contact_count),
            "episode_seconds_target": self.episode_seconds,
            "max_episode_actions": int(self.episode_steps),
            "eva_attempts": int(self.total_eva_attempts),
            "valid_eva_casts": int(self.total_valid_eva_casts),
            "invalid_eva_attempts": int(self.total_invalid_eva_attempts),
            "missed_eva_opportunities": int(self.total_missed_eva_opportunities),
            "valid_eva_rate": float(
                self.total_valid_eva_casts / max(1, self.total_eva_attempts)
            ),
            "kills_per_valid_eva": float(
                self.total_kills / max(1, self.total_valid_eva_casts)
            ),
            "kills_per_simulated_hour": float(
                self.total_kills * 3600.0 / max(1.0e-9, self.elapsed)
            ),
            "approach_potential_cells": float(self._approach_potential_cells),
            "best_group_actor_id": self._best_group_actor_id,
            "reward_components": reward_components,
            "reward_component_totals": dict(self.reward_component_totals),
            "reward_contract": self.reward_calculator.config.as_dict(),
        }

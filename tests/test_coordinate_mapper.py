from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from libs.HumanKeyboard import KeyPressTiming
from mapper import Mapper
from mapper.CoordinateFrame import CoordinateFrame
from mapper.CoordinateMapper import (
    CoordinateMapper,
    MapperConfig,
    MapProfileLocationMismatch,
    MapTransitionDetected,
    NativeMotion,
    load_mapper_config,
)
from mapper.MinimapHeading import HeadingReading
from mapper.OccupancyGrid import BLOCKED, FREE, OccupancyGrid
from position import PlayerPose
from runtime.worker_manager import CancellationToken


def _pose(x: float, y: float, z: float) -> PlayerPose:
    return PlayerPose(x=x, y=y, z=z, heading_degrees=None, timestamp=1.0)


class _FakeController:
    def __init__(self) -> None:
        self.forward_calls: list[float] = []
        self.backward_calls: list[float] = []
        self.left_calls: list[float] = []
        self.right_calls: list[float] = []
        self.settle_calls: list[float] = []

    @staticmethod
    def _timing(seconds: float) -> KeyPressTiming:
        return KeyPressTiming(
            requested_seconds=seconds,
            clamped_seconds=seconds,
            held_seconds=seconds,
            elapsed_seconds=seconds,
        )

    def forward(self, seconds: float) -> KeyPressTiming:
        self.forward_calls.append(seconds)
        return self._timing(seconds)

    def backward(self, seconds: float) -> KeyPressTiming:
        self.backward_calls.append(seconds)
        return self._timing(seconds)

    def turn_left(self, seconds: float) -> KeyPressTiming:
        self.left_calls.append(seconds)
        return self._timing(seconds)

    def turn_right(self, seconds: float) -> KeyPressTiming:
        self.right_calls.append(seconds)
        return self._timing(seconds)

    def settle(self, seconds: float) -> None:
        self.settle_calls.append(seconds)

    def stop(self) -> None:
        return None


class _FakeBot:
    def __init__(self, poses: list[PlayerPose]) -> None:
        self.keyboard = SimpleNamespace()
        self._poses = deque(poses)

    def get_player_pose(self) -> PlayerPose | None:
        if not self._poses:
            raise AssertionError("No fake poses remain")
        return self._poses.popleft()


class _FakeLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write(self, row: dict[str, object]) -> None:
        self.rows.append(row)

    def close(self) -> None:
        return None


def _bare_mapper() -> CoordinateMapper:
    mapper = CoordinateMapper.__new__(CoordinateMapper)
    mapper.config = MapperConfig()
    mapper.controller = _FakeController()
    mapper.grid = OccupancyGrid()
    mapper.coordinate_frame = CoordinateFrame(
        origin_native_x=100.0,
        origin_native_z=200.0,
        native_units_per_cell=1.6,
    )
    mapper._heading_source = "unknown"
    mapper._blocked_counts = {}
    mapper._last_pose = None
    mapper._trusted_pose = None
    mapper._trap_score = 0
    mapper._trap_entry_edge = None
    mapper._recent_traversed_edges = []
    mapper._step = 1
    mapper._post_eva_guard_pending = False
    mapper._focus_pause_announced = False
    mapper.cancellation = CancellationToken()
    mapper.status_callback = lambda _message: None
    mapper.logger = _FakeLogger()
    mapper.map_profile = SimpleNamespace(name="Test map")
    return mapper




def test_coordinate_mapper_config_loads_from_json(tmp_path: Path) -> None:
    path = tmp_path / "coordinate_mapper.json"
    path.write_text(
        '{"version": 1, "grid_size": 501, "blocked_distance_units": 0.2, '
        '"teleport_vertical_distance_units": 31.5}',
        encoding="utf-8",
    )

    config = load_mapper_config(path)

    assert config.grid_size == 501
    assert config.blocked_distance_units == pytest.approx(0.2)
    assert config.teleport_vertical_distance_units == pytest.approx(31.5)


def test_shipped_coordinate_mapper_config_loads_without_schema_mismatch() -> None:
    config_path = Path(__file__).parents[1] / "mapper" / "coordinate_mapper.json"

    config = load_mapper_config(config_path)

    assert config.teleport_distance_units == pytest.approx(50.0)
    assert config.teleport_vertical_distance_units == pytest.approx(25.0)
    assert config.eva_settle_seconds == pytest.approx(0.40)
    assert config.eva_retry_settle_seconds == pytest.approx(0.25)
    assert config.slide_min_forward_alignment == pytest.approx(0.70)
    assert config.pause_when_unfocused is True
    assert config.backward_seconds == pytest.approx(0.18)
    assert config.backward_retry_seconds == pytest.approx(0.28)
    assert config.backward_blocked_distance_units == pytest.approx(0.05)
    assert config.backward_partial_distance_units == pytest.approx(0.35)
    assert config.minimap_heading_enabled is True
    assert config.heading_collision_guard_degrees == pytest.approx(18.0)
    assert config.free_space_autofill_enabled is True
    # The shipped profile deliberately applies the conservative 1x1 override
    # instead of the broader dataclass defaults.
    assert config.free_space_autofill_max_enclosed_area_cells == 1
    assert config.free_space_autofill_max_enclosed_span_cells == 1
    assert config.free_space_autofill_max_line_length_cells == 40

def test_runtime_mapper_alias_uses_coordinate_mapper() -> None:
    assert Mapper is CoordinateMapper


def test_coordinate_frame_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "coordinate_frame.json"
    frame = CoordinateFrame(
        origin_native_x=123.5,
        origin_native_z=-80.25,
        native_units_per_cell=1.6,
    )
    frame.save(path)

    loaded = CoordinateFrame.load(path)

    assert loaded == frame
    assert loaded.to_local_cells(126.7, -77.05) == pytest.approx((2.0, 2.0))


@pytest.mark.parametrize(
    ("delta_x", "delta_z", "expected"),
    [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 90.0),
        (0.0, -1.0, 180.0),
        (-1.0, 0.0, 270.0),
        (1.0, 1.0, 45.0),
    ],
)
def test_heading_is_derived_from_native_displacement(
    delta_x: float,
    delta_z: float,
    expected: float,
) -> None:
    assert CoordinateMapper.heading_from_delta(delta_x, delta_z) == pytest.approx(expected)


def test_native_forward_classification_uses_only_coordinate_distance() -> None:
    mapper = _bare_mapper()
    mapper.bot = _FakeBot(
        [
            _pose(10.0, 5.0, 20.0),
            _pose(10.04, 5.0, 20.0),
            _pose(10.0, 5.0, 20.0),
            _pose(11.6, 5.0, 20.0),
            _pose(10.0, 5.0, 20.0),
            _pose(110.0, 5.0, 20.0),
        ]
    )

    blocked, _ = mapper._measure_forward()
    moved, _ = mapper._measure_forward()
    teleport, _ = mapper._measure_forward()

    assert blocked.outcome == "blocked"
    assert blocked.horizontal_distance == pytest.approx(0.04)
    assert moved.outcome == "moved"
    assert moved.horizontal_distance == pytest.approx(1.6)
    assert teleport.outcome == "teleport"


def test_integrate_motion_draws_native_segment_and_corrects_heading() -> None:
    mapper = _bare_mapper()
    mapper.grid.add_contact_boundary(
        from_x=0, from_y=0, to_x=1, to_y=0, heading_deg=90.0, confirmations=2
    )
    motion = NativeMotion(
        before=_pose(100.0, 50.0, 200.0),
        after=_pose(103.2, 50.0, 200.0),
        delta_x=3.2,
        delta_y=0.0,
        delta_z=0.0,
        horizontal_distance=3.2,
        vertical_distance=0.0,
        outcome="moved",
    )

    heading = mapper._integrate_motion(motion)

    assert heading == pytest.approx(90.0)
    assert mapper.grid.continuous_pose.x == pytest.approx(2.0)
    assert mapper.grid.continuous_pose.y == pytest.approx(0.0)
    assert mapper.grid.value(0, 0) == FREE
    assert mapper.grid.value(1, 0) == FREE
    assert mapper.grid.value(2, 0) == FREE
    assert not mapper.grid.contact_boundary_blocks(0, 0, 1, 0)
    assert mapper._heading_source == "movement"


def test_turn_is_one_pulse_and_leaves_heading_unverified_without_minimap() -> None:
    mapper = _bare_mapper()
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    mapper._trusted_pose = _pose(100.0, 50.0, 200.0)
    mapper.bot = _FakeBot([_pose(100.0, 50.0, 200.0)])

    mapper._execute_turn(left=True, reason="frontier")

    assert mapper.controller.left_calls == [mapper.config.turn_left_90_seconds]
    assert mapper.controller.right_calls == []
    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(0.0)
    assert mapper._heading_source == "commanded-unverified"
    assert mapper.grid.metadata.heading_known is False


def test_blocked_boundary_requires_two_native_zero_movement_confirmations() -> None:
    mapper = _bare_mapper()
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    pose = _pose(100.0, 50.0, 200.0)

    edge = mapper._intended_edge(pose)
    first_count, first_confirmed = mapper._record_blocked_boundary(edge)

    assert first_count == 1
    assert first_confirmed is False
    assert mapper.grid.metadata.contact_boundaries == []

    second_count, second_confirmed = mapper._record_blocked_boundary(edge)

    assert second_count == 2
    assert second_confirmed is True
    assert edge == ((0, 0), (1, 0))
    assert mapper.grid.contact_boundary_blocks(0, 0, 1, 0)
    assert mapper.grid.value(1, 0) == BLOCKED


def test_execute_forward_confirms_block_from_two_native_attempts() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.0, 50.0, 200.0),
            _pose(100.02, 50.0, 200.0),
            _pose(100.02, 50.0, 200.0),
            _pose(100.04, 50.0, 200.0),
        ]
    )

    mapper._execute_forward("unknown neighbor")

    assert mapper.grid.contact_boundary_blocks(0, 0, 1, 0)
    assert len(mapper.logger.rows) == 2
    assert mapper.logger.rows[-1]["note"] == (
        "blocked boundary confirmed from native displacement"
    )


def test_partial_motion_is_classified_separately() -> None:
    mapper = _bare_mapper()
    mapper.bot = _FakeBot(
        [
            _pose(10.0, 5.0, 20.0),
            _pose(10.4, 5.0, 20.0),
        ]
    )

    motion, _ = mapper._measure_forward()

    assert motion.outcome == "partial"
    assert motion.horizontal_distance == pytest.approx(0.4)


def test_turn_duration_scales_from_measured_non_cardinal_heading() -> None:
    mapper = _bare_mapper()
    mapper.grid.set_continuous_pose(0.0, 0.0, 70.0)
    mapper._trusted_pose = _pose(100.0, 50.0, 200.0)
    mapper.bot = _FakeBot([_pose(100.0, 50.0, 200.0)])

    mapper._execute_turn(left=True, reason="frontier")

    expected = mapper.config.turn_left_90_seconds * 70.0 / 90.0
    assert mapper.controller.left_calls == [pytest.approx(expected)]
    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(0.0)


def test_cross_action_teleport_is_detected_before_next_command() -> None:
    mapper = _bare_mapper()
    mapper._trusted_pose = _pose(256.727753, 101.3, 72.120857)
    mapper.bot = _FakeBot([_pose(6980.0, 101.3, 3309.0)])

    with pytest.raises(MapTransitionDetected) as caught:
        mapper._verify_pose_continuity("between mapper actions")

    assert caught.value.horizontal_distance > 7000.0
    assert mapper._trusted_pose.x == pytest.approx(256.727753)


def test_turn_detects_transition_that_occurs_during_settle() -> None:
    mapper = _bare_mapper()
    mapper.grid.set_continuous_pose(2.33, -8.67, 174.0)
    mapper._trusted_pose = _pose(256.727753, 101.3, 72.120857)
    mapper.bot = _FakeBot([_pose(6980.0, 101.3, 3309.0)])

    with pytest.raises(MapTransitionDetected):
        mapper._execute_turn(left=True, reason="frontier")

    # The destination was never accepted as the current map pose.
    assert mapper.grid.continuous_pose.x == pytest.approx(2.33)
    assert mapper.grid.continuous_pose.y == pytest.approx(-8.67)


def test_vertical_impossible_movement_is_also_a_transition() -> None:
    mapper = _bare_mapper()
    mapper.bot = _FakeBot(
        [
            _pose(10.0, 5.0, 20.0),
            _pose(10.0, 40.0, 20.0),
        ]
    )

    motion, _ = mapper._measure_forward()

    assert motion.outcome == "teleport"
    assert motion.vertical_distance == pytest.approx(35.0)


def test_transition_record_preserves_last_trusted_map_pose() -> None:
    mapper = _bare_mapper()
    mapper.grid.set_continuous_pose(2.33, -8.67, 174.0)
    transition = MapTransitionDetected(
        before=_pose(256.727753, 101.3, 72.120857),
        after=_pose(6980.0, 101.3, 3309.0),
        horizontal_distance=7460.0,
        vertical_distance=0.0,
        context="between mapper actions",
    )

    mapper._record_map_transition(transition)

    assert mapper.grid.continuous_pose.x == pytest.approx(2.33)
    assert mapper.grid.continuous_pose.y == pytest.approx(-8.67)
    assert mapper.grid.metadata.position_known is False
    assert len(mapper.grid.metadata.suspected_transitions) == 1
    assert mapper.logger.rows[-1]["action"] == "MAP_TRANSITION"
    assert mapper.logger.rows[-1]["motion_outcome"] == "teleport"


def test_startup_rejects_coordinates_outside_existing_map_frame() -> None:
    mapper = _bare_mapper()

    with pytest.raises(MapProfileLocationMismatch) as caught:
        mapper._validate_start_location(_pose(6980.0, 101.3, 3309.0))

    assert caught.value.local_x > 4000.0
    assert caught.value.local_y > 1900.0


def test_blocked_confirmation_keeps_first_edge_across_rounding_drift() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.49, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.784, 50.0, 200.0),
            _pose(100.816, 50.0, 200.0),
            _pose(100.816, 50.0, 200.0),
            _pose(100.848, 50.0, 200.0),
        ]
    )

    mapper._execute_forward("unknown neighbor")

    boundaries = mapper.grid.metadata.contact_boundaries
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert (boundary["from_x"], boundary["from_y"]) == (0, 0)
    assert (boundary["to_x"], boundary["to_y"]) == (0, 1)


def test_repeated_partial_progress_triggers_backward_escape() -> None:
    mapper = _bare_mapper()
    mapper.config = MapperConfig(
        trap_score_threshold=2,
        partial_trap_score=2,
        escape_backward_steps=1,
        escape_target_distance_units=0.5,
    )
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper._recent_traversed_edges = [((0, -1), (0, 0))]
    mapper.bot = _FakeBot(
        [
            # Partial forward.
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 200.4),
            # Backward escape.
            _pose(100.0, 50.0, 200.4),
            _pose(100.0, 50.0, 199.4),
        ]
    )

    mapper._execute_forward("unknown neighbor")

    assert mapper.controller.backward_calls == [mapper.config.backward_seconds]
    assert mapper.grid.contact_boundary_blocks(0, -1, 0, 0)
    assert mapper._trap_score == 0


def test_slow_backpedal_uses_backward_thresholds_instead_of_forward_thresholds() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 199.92),
        ]
    )

    motion, _timing = mapper._measure_backward()

    # 0.08u would be blocked by the 0.10u forward threshold, but it is valid
    # slow backpedal progress under the separate 0.05u reverse threshold.
    assert motion.horizontal_distance == pytest.approx(0.08)
    assert motion.outcome == "partial"
    assert mapper.controller.backward_calls == [mapper.config.backward_seconds]


def test_backward_escape_rechecks_slow_start_before_declaring_it_blocked() -> None:
    mapper = _bare_mapper()
    mapper.config = MapperConfig(
        escape_backward_steps=1,
        escape_target_distance_units=0.20,
        backward_seconds=0.18,
        backward_retry_seconds=0.28,
        backward_blocked_distance_units=0.05,
        backward_partial_distance_units=0.35,
    )
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            # Normal backpedal pulse starts slowly and looks blocked.
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 199.98),
            # Longer recheck proves that reverse movement is available.
            _pose(100.0, 50.0, 199.98),
            _pose(100.0, 50.0, 199.70),
        ]
    )
    mapper._trap_entry_edge = None

    mapper._recover_from_trap("test slow reverse")

    assert mapper.controller.backward_calls == [0.18, 0.28]
    assert mapper.grid.cells[mapper.grid.cells == BLOCKED].size == 0
    assert not mapper.grid.metadata.contact_boundaries
    assert mapper.grid.continuous_pose.y < 0.0


def test_confirmed_failed_backpedal_marks_rear_boundary_blocked() -> None:
    mapper = _bare_mapper()
    mapper.config = MapperConfig(
        escape_backward_steps=1,
        backward_blocked_confirmations=1,
        escape_wiggle_after_blocked_attempts=1,
    )
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 200.0),
        ]
    )
    mapper._trap_entry_edge = None
    mapper._escape_wiggle = lambda **_kwargs: None  # type: ignore[method-assign]
    mapper._radial_escape_probe = lambda _reason: False  # type: ignore[method-assign]

    mapper._recover_from_trap("test blocked reverse")

    assert mapper.controller.backward_calls == [
        mapper.config.backward_seconds,
        mapper.config.backward_retry_seconds,
    ]
    assert mapper.grid.value(0, -1) == BLOCKED
    assert mapper.grid.contact_boundary_blocks(0, 0, 0, -1)
    assert mapper.grid.metadata.contact_boundaries[-1]["heading_deg"] == pytest.approx(180.0)


def test_coordinate_frame_skips_replacing_identical_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "coordinate_frame.json"
    frame = CoordinateFrame(
        origin_native_x=123.5,
        origin_native_z=-80.25,
        native_units_per_cell=1.6,
    )
    frame.save(path)

    def unexpected_replace(_source: object, _destination: object) -> None:
        raise AssertionError("identical immutable frame should not be replaced")

    monkeypatch.setattr("mapper.CoordinateFrame.os.replace", unexpected_replace)

    frame.save(path)

    assert CoordinateFrame.load(path) == frame
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_coordinate_frame_retries_transient_windows_replace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "coordinate_frame.json"
    frame = CoordinateFrame(
        origin_native_x=10.0,
        origin_native_z=20.0,
        native_units_per_cell=1.6,
    )
    real_replace = __import__("os").replace
    attempts = 0

    def flaky_replace(source: object, destination: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "destination temporarily locked")
        real_replace(source, destination)

    monkeypatch.setattr("mapper.CoordinateFrame.os.replace", flaky_replace)
    monkeypatch.setattr("mapper.CoordinateFrame.time.sleep", lambda _seconds: None)

    frame.save(path)

    assert attempts == 3
    assert CoordinateFrame.load(path) == frame
    assert list(tmp_path.glob(".*.tmp")) == []


def test_mapper_checkpoint_lock_does_not_stop_mapping(tmp_path: Path) -> None:
    mapper = _bare_mapper()
    mapper.frame_path = tmp_path / "persistent" / "coordinate_frame.json"
    mapper.map_dir = tmp_path / "persistent"
    mapper.output_dir = tmp_path / "run"
    mapper._save_failure_streak = 0
    messages: list[str] = []
    mapper.status_callback = messages.append

    class FlakyFrame:
        def __init__(self) -> None:
            self.calls = 0

        def save(self, path: Path) -> None:
            self.calls += 1
            if self.calls == 1:
                raise PermissionError(5, "temporarily locked", str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    mapper.coordinate_frame = FlakyFrame()

    mapper._save_state()

    assert mapper._save_failure_streak == 1
    assert any("mapping continues" in message for message in messages)
    assert (mapper.output_dir / "coordinate_frame.json").is_file()


def test_mapper_checkpoint_reports_recovery_after_deferred_save(tmp_path: Path) -> None:
    mapper = _bare_mapper()
    mapper.frame_path = tmp_path / "persistent" / "coordinate_frame.json"
    mapper.map_dir = tmp_path / "persistent"
    mapper.output_dir = tmp_path / "run"
    mapper._save_failure_streak = 0
    messages: list[str] = []
    mapper.status_callback = messages.append

    class RecoveringFrame:
        def __init__(self) -> None:
            self.locked = True

        def save(self, path: Path) -> None:
            if self.locked:
                raise PermissionError(5, "temporarily locked", str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    frame = RecoveringFrame()
    mapper.coordinate_frame = frame
    mapper._save_state()
    frame.locked = False
    mapper._save_state()

    assert mapper._save_failure_streak == 0
    assert any("checkpoint recovered" in message for message in messages)


def test_coordinate_mapper_config_accepts_local_map_radius(tmp_path: Path) -> None:
    path = tmp_path / "coordinate_mapper.json"
    path.write_text(
        '{"version": 1, "local_map_radius_cells": 17}',
        encoding="utf-8",
    )

    config = load_mapper_config(path)

    assert config.local_map_radius_cells == 17


def test_publish_map_uses_combined_dashboard() -> None:
    mapper = _bare_mapper()
    frames = []
    mapper.frame_callback = frames.append
    mapper._last_publish_at = 0.0
    mapper.config = MapperConfig(local_map_radius_cells=9)

    mapper._publish_map(force=True)

    assert len(frames) == 1
    dashboard = frames[0]
    assert dashboard.ndim == 3
    assert dashboard.shape[1] > dashboard.shape[0]



def test_lateral_slide_marks_intended_boundary_blocked_and_preserves_heading() -> None:
    mapper = _bare_mapper()
    mapper.config = MapperConfig(heading_mismatch_recheck_degrees=90.0)
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.0, 50.0, 200.0),
            # Mostly east while the commanded direction is north.
            _pose(101.0, 50.0, 200.25),
        ]
    )

    mapper._execute_forward("unknown neighbor")

    assert mapper.grid.contact_boundary_blocks(0, 0, 0, 1)
    assert mapper.grid.value(0, 1) == BLOCKED
    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(0.0)
    assert mapper.grid.continuous_pose.x == pytest.approx(0.625)
    assert mapper.grid.continuous_pose.y == pytest.approx(0.15625)
    assert mapper.logger.rows[-1]["motion_outcome"] == "sliding"


def test_first_failed_move_after_eva_is_retried_without_wall_evidence() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "commanded"
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    mapper._post_eva_guard_pending = True
    mapper.bot = _FakeBot(
        [
            # Cast lock: no movement.
            _pose(100.0, 50.0, 200.0),
            _pose(100.0, 50.0, 200.0),
            # Retry succeeds.
            _pose(100.0, 50.0, 200.0),
            _pose(101.6, 50.0, 200.0),
        ]
    )

    mapper._execute_forward("unknown neighbor")

    assert mapper.grid.value(1, 0) == FREE
    assert not mapper.grid.contact_boundary_blocks(0, 0, 1, 0)
    assert mapper.controller.forward_calls == [
        mapper.config.forward_seconds,
        mapper.config.forward_seconds,
    ]
    assert mapper.config.eva_retry_settle_seconds in mapper.controller.settle_calls
    assert any(
        row.get("note") == "post-EVA movement result ignored as collision evidence"
        for row in mapper.logger.rows
    )


def test_unfocused_mapper_waits_instead_of_sending_movement() -> None:
    mapper = _bare_mapper()
    messages: list[str] = []
    mapper.status_callback = messages.append

    class FocusKeyboard:
        def __init__(self) -> None:
            self.states = iter((False, False, True))

        def is_target_foreground(self) -> bool:
            return next(self.states)

    class ImmediateToken:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait(self, timeout: float) -> bool:
            self.waits.append(timeout)
            return False

        def raise_if_cancelled(self) -> None:
            return None

    token = ImmediateToken()
    mapper.bot = SimpleNamespace(keyboard=FocusKeyboard())
    mapper.cancellation = token

    mapper._wait_for_game_focus()

    assert token.waits == [mapper.config.unfocused_poll_seconds]
    assert any("mapping is paused" in message for message in messages)
    assert messages[-1] == "FlyFF regained focus; coordinate mapping is resuming."
    assert mapper.controller.settle_calls[-1] == mapper.config.settle_seconds


def _heading_reading(angle: float) -> HeadingReading:
    return HeadingReading(
        angle_deg=angle,
        confidence=0.95,
        center=(0, 0),
        radius=8,
        angular_uncertainty_deg=1.0,
        ambiguity=0.1,
    )


def test_initial_heading_uses_minimap_without_movement_probe() -> None:
    mapper = _bare_mapper()
    mapper._read_minimap_heading = (  # type: ignore[method-assign]
        lambda _context: _heading_reading(123.0)
    )
    mapper._publish_map = lambda **_kwargs: None  # type: ignore[method-assign]

    mapper._acquire_heading()

    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(123.0)
    assert mapper._heading_source == "minimap"
    assert mapper.grid.metadata.heading_known is True
    assert mapper.controller.forward_calls == []


def test_native_displacement_refines_minimap_heading_precisely() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "minimap"
    mapper.grid.set_continuous_pose(0.0, 0.0, 40.0)
    motion = NativeMotion(
        before=_pose(100.0, 50.0, 200.0),
        after=_pose(101.0, 50.0, 201.0),
        delta_x=1.0,
        delta_y=0.0,
        delta_z=1.0,
        horizontal_distance=2**0.5,
        vertical_distance=0.0,
        outcome="moved",
        forward_progress=2**0.5,
        lateral_distance=0.0,
        forward_alignment=1.0,
    )

    heading = mapper._integrate_motion(motion, infer_heading=True)

    assert heading == pytest.approx(45.0)
    assert mapper._heading_source == "movement"


def test_partial_motion_does_not_validate_unverified_turn_heading() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "commanded-unverified"
    mapper.grid.set_continuous_pose(0.0, 0.0, 90.0)
    motion = NativeMotion(
        before=_pose(100.0, 50.0, 200.0),
        after=_pose(100.2, 50.0, 200.0),
        delta_x=0.2,
        delta_y=0.0,
        delta_z=0.0,
        horizontal_distance=0.2,
        vertical_distance=0.0,
        outcome="partial",
        forward_progress=0.2,
        lateral_distance=0.0,
        forward_alignment=1.0,
    )

    mapper._integrate_motion(motion, infer_heading=False)

    assert mapper._heading_source == "uncertain"
    assert mapper.grid.metadata.heading_known is False


def test_minimap_correction_prevents_false_slide_blocker() -> None:
    mapper = _bare_mapper()
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.0, 50.0, 200.0),
            _pose(101.0, 50.0, 200.0),
        ]
    )
    mapper._read_minimap_heading = (  # type: ignore[method-assign]
        lambda _context: _heading_reading(90.0)
    )

    mapper._execute_forward("unknown neighbor")

    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(90.0)
    assert mapper.grid.value(1, 0) == FREE
    assert not mapper.grid.metadata.contact_boundaries


def test_verified_backpedal_slide_marks_rear_boundary_and_actual_path() -> None:
    mapper = _bare_mapper()
    mapper.config = MapperConfig(
        escape_backward_steps=1,
        heading_mismatch_recheck_degrees=90.0,
    )
    mapper._heading_source = "movement"
    mapper.grid.set_continuous_pose(0.0, 0.0, 0.0)
    mapper.bot = _FakeBot(
        [
            _pose(100.0, 50.0, 200.0),
            # Backward command points south, but collision causes an eastward slide.
            _pose(101.0, 50.0, 199.8),
        ]
    )
    mapper._trap_entry_edge = None

    mapper._recover_from_trap("test reverse slide")

    assert mapper.grid.contact_boundary_blocks(0, 0, 0, -1)
    assert mapper.grid.value(0, -1) == BLOCKED
    assert mapper.grid.value(1, 0) == FREE
    assert mapper.grid.continuous_pose.heading_deg == pytest.approx(0.0)

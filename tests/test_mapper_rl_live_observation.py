from __future__ import annotations

from mapper.AdaptiveMotionModel import AdaptiveForwardOutcome
from mapper.AdaptiveMotionTracker import DirectionalFlow, MotionEstimate
from mapper.OccupancyGrid import OccupancyGrid
from mapper.rl.LiveObservation import (
    LivePolicyMemory,
    build_live_observation,
    build_live_policy_input,
)
from mapper.rl.PolicyTypes import ObservationQuality


def test_live_observation_uses_persistent_grid() -> None:
    grid = OccupancyGrid(size=41)
    grid.set_continuous_pose(0.0, 0.0, 0.0)
    grid.mark_free(0, 1)
    grid.mark_blocked(1, 1)
    grid.set_pose_reliability(
        position_known=True,
        heading_known=True,
        note="test",
    )

    observation = build_live_observation(grid, LivePolicyMemory())

    assert observation["local_map"].shape[0] == 5
    assert observation["state"].ndim == 1


def test_low_tracks_and_large_change_are_camera_obscured() -> None:
    memory = LivePolicyMemory()
    motion = MotionEstimate(
        change_score=0.22,
        teleport_likely=False,
        directional_flow=DirectionalFlow(
            scene_dx_px=0.0,
            scene_dy_px=0.0,
            magnitude_px=0.0,
            dispersion_px=0.0,
            tracked_points=2,
            inlier_ratio=0.0,
            confidence=0.0,
            detected_points=100,
            valid_tracks=2,
        ),
    )

    memory.update_after_step(
        actual_action="FORWARD",
        forward_outcome=AdaptiveForwardOutcome.UNCERTAIN,
        motion=motion,
        pose_known=False,
        heading_known=True,
    )

    assert memory.quality is ObservationQuality.CAMERA_OBSCURED


def test_live_policy_input_includes_safe_action_mask() -> None:
    grid = OccupancyGrid(size=41)
    grid.set_continuous_pose(0.0, 0.0, 0.0)
    grid.set_pose_reliability(position_known=True, heading_known=True, note="test")
    memory = LivePolicyMemory()

    policy_input = build_live_policy_input(grid, memory)

    assert policy_input.action_mask.shape == (6,)
    assert policy_input.action_mask.any()
    assert policy_input.observation["state"].shape[0] > 30


def test_live_wait_streak_is_tracked_for_shadow_action_masks() -> None:
    memory = LivePolicyMemory()

    memory.update_after_step(
        actual_action="WAIT",
        forward_outcome=None,
        motion=None,
        pose_known=True,
        heading_known=True,
    )
    memory.update_after_step(
        actual_action="WAIT",
        forward_outcome=None,
        motion=None,
        pose_known=True,
        heading_known=True,
    )

    assert memory.wait_streak == 2
    memory.update_after_step(
        actual_action="FORWARD",
        forward_outcome=AdaptiveForwardOutcome.MOVED,
        motion=None,
        pose_known=True,
        heading_known=True,
    )
    assert memory.wait_streak == 0

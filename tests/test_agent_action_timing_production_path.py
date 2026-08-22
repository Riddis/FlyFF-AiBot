"""Production-path regression test for the Agent recording action-timing
fix (MISTAKES.md; docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md
section 1d).

Before this fix, ``farming.trainer.run_native_farming_agent`` published
its runtime ``"action"`` event AFTER ``gym.step()`` returned, so
``RecordingSink`` learned about an action only once it had already
finished executing -- frames sampled by the sink's own poll thread WHILE
a ``gym.step()`` call was still running carried the *previous* action.
For momentary EVA/jump commands (pressed and released entirely inside a
single ``gym.step()`` call) this meant those frames were never labeled
correctly, and a final action issued immediately before an
``"episode_end"`` reset could disappear from ``frame.action`` entirely.

Unlike ``tests/test_recording_action_presence_provenance.py``'s
``test_control_actions_reach_frame_action_and_survive_world_model_
fitting`` (which injects ``"action"`` events directly via
``RecordingSink.add_runtime_event`` and therefore cannot see a timing bug
in ``run_native_farming_agent`` itself), this test drives the REAL
``run_native_farming_agent`` control loop end to end:

    run_native_farming_agent -> (fake) gym.step() -> on_runtime_event
        -> RecordingSink.add_runtime_event -> RecordingArchive

A fake gym environment blocks each ``step()`` call until it can prove --
by watching ``RecordingSink``'s own real frame counter increase, not a
fixed sleep -- that the sink's real poll thread sampled at least one
frame while that specific call was still executing. This reproduces the
exact race the bug depended on, deterministically."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bot.recording_sink import RecordingOwnership, RecordingSink, build_runtime_metadata
from farming.actions import FarmingCommand, FarmingEvent, SteeringAction
from farming.config import FarmingRuntimeConfig
from farming.trainer import run_native_farming_agent
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
)
from runtime.worker_manager import CancellationToken
from simulator.map_model import MapModel
from simulator.schema import RecordingArchive
from simulator.world_model import fit_world_model


class _FakeNativeProcessService:
    """Matches the surface RecordingSink/build_runtime_metadata need --
    see tests/test_recording_action_presence_provenance.py::_FakeService,
    duplicated locally so this test stays self-contained."""

    def __init__(self) -> None:
        self.attach_policy = None
        self.presence_validation_source = "authoritative_refresh"
        self.presence_species_validated = True
        self.recovered_presence_species_offset = 0x1000

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        return NativePointerSnapshot(
            player_pointer_address=0x500000,
            world_pointer_address=0x600000,
            player_base=0x20000000,
            world_base=0x30000000,
            generation=1,
            captured_at=time.monotonic(),
        )


class _FakePositionProvider:
    def __init__(self) -> None:
        self.x = 0.0
        self.z = 0.0

    def read_pose(self, *, pointer_snapshot=None):
        from position.PositionProvider import PlayerPose

        self.x += 0.1
        return PlayerPose(
            x=self.x, y=0.0, z=self.z, heading_degrees=0.0, timestamp=time.monotonic()
        )


class _FakeMonsterProvider:
    def refresh_slot_cache(self, pointer_snapshot, *, cancellation=None, deadline=None, force=False):
        return ActorCacheRefreshResult(
            ActorCacheOutcome.REFRESHED, pointer_snapshot.world_base, pointer_snapshot.generation, slot_count=0
        )

    def read_cached_active_actors(self, pointer_snapshot, player_pose, *, allowed_species_ids=None, vision_radius_native=None):
        return CachedActorReadResult(
            ActorCacheOutcome.READY, pointer_snapshot.world_base, pointer_snapshot.generation, actors=()
        )

    def read_actor_hp_states(self, candidates):
        return {}


class _FakeBot:
    """Only the surface run_native_farming_agent actually touches."""

    def __init__(self) -> None:
        self.rl_enabled = False

    def start(self) -> None:
        self.rl_enabled = True

    def read_penya_count(self) -> int | None:
        return None


class _ScriptedModel:
    """Stand-in for the loaded SessionAwarePPO: replays a fixed
    factorized [steering, event] action sequence, ignoring the
    observation (the fake gym below produces meaningless observations --
    only the control-loop/event-ordering plumbing is under test here)."""

    def __init__(self, actions: list[tuple[int, int]]) -> None:
        self._actions = actions
        self.calls = 0
        self.num_timesteps = 0

    def predict(self, observation, deterministic: bool = True):
        # A real policy's inference takes non-negligible time. This gap
        # is what lets the sink's poll thread sample a frame in the
        # "idle" window between one step's action-result/revert events
        # and the next step's action-publish event -- without it, a fast
        # fake loop could race straight through that window and this
        # test would only ever observe frames sampled inside a
        # gym.step() call, never the between-steps state.
        time.sleep(0.03)
        steering, event = self._actions[self.calls]
        self.calls += 1
        return np.array([steering, event], dtype=np.int64), None


class _BarrierGym:
    """Fake ``runtime.gym``: each ``step()`` call blocks until the real
    RecordingSink's poll thread has demonstrably written at least one
    frame while this exact call is still executing -- proven via the
    sink's own frame counter increasing, not a fixed sleep -- then
    records which frame sequence numbers were sampled during that
    window before returning the scripted step result."""

    def __init__(self, sink: RecordingSink, steps: list[dict[str, object]]) -> None:
        self._sink = sink
        self._steps = steps
        self._index = 0
        self.sampled_during: dict[int, range] = {}

    def reset(self):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, factorized_action):
        index = self._index
        self._index += 1
        spec = self._steps[index]

        frame_count_before = self._sink._frame_count
        deadline = time.monotonic() + 5.0
        while True:
            frame_count_now = self._sink._frame_count
            if frame_count_now > frame_count_before:
                self.sampled_during[index] = range(frame_count_before, frame_count_now)
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"No recording frame was sampled while the fake "
                    f"gym.step() for step {index} ({spec['label']!r}) "
                    "was executing -- the action-timing fix did not "
                    "publish the action before calling gym.step()."
                )
            time.sleep(0.002)

        return (
            np.zeros(4, dtype=np.float32),
            spec["reward"],
            spec["terminated"],
            False,
            spec["info"],
        )


def _legacy_action(steering: int, event: int) -> int:
    return int(FarmingCommand(SteeringAction(steering), FarmingEvent(event)).legacy_action)


def _make_sink(tmp_path: Path) -> RecordingSink:
    service = _FakeNativeProcessService()
    metadata = build_runtime_metadata(
        SimpleNamespace(config={}),
        attach_policy_name="STANDARD",
        presence_validation_source=service.presence_validation_source,
        presence_species_validated=service.presence_species_validated,
        presence_species_offset=service.recovered_presence_species_offset,
    )
    return RecordingSink(
        native_process_service=service,
        position_provider=_FakePositionProvider(),
        monster_provider=_FakeMonsterProvider(),
        ownership=RecordingOwnership(started_by="RUNTIME_AUTO"),
        character_name="Agent Timing Character",
        frame_interval_seconds=0.005,
        output_root=tmp_path,
        metadata=metadata,
    )


def test_run_native_farming_agent_publishes_action_before_gym_step_executes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex's reproduced failure sequence, driven through the real
    production Agent path end to end, must now resolve correctly:

        [1,0]  steering-only LEFT             -> during-step label 1
        [2,1]  RIGHT + CAST_EVA               -> during-step label 3
        [1,2]  LEFT + JUMP                    -> during-step label 4
        [2,0]  steering-only RIGHT            -> during-step label 2
        [0,0]  final STRAIGHT, then terminate -> during-step label 0

    plus: an initial no-action interval reads -1, transient EVA/jump
    frames are never mislabeled onto the neighboring step, and
    episode_end only clears the action after the final step's frames
    were already sampled."""

    steps = [
        {"label": "steering_left", "steering": 1, "event": 0, "reward": 0.1, "terminated": False},
        {"label": "eva_from_right", "steering": 2, "event": 1, "reward": 0.2, "terminated": False},
        {"label": "jump_from_left", "steering": 1, "event": 2, "reward": 0.3, "terminated": False},
        {"label": "steering_right", "steering": 2, "event": 0, "reward": 0.1, "terminated": False},
        {"label": "final_straight", "steering": 0, "event": 0, "reward": 0.5, "terminated": True},
    ]
    for step in steps:
        step["info"] = {
            "native_kill_delta": 0,
            "action_name": "RUN_FORWARD",
            "jump_requested": step["event"] == 2,
            "jump_performed": step["event"] == 2,
        }

    sink = _make_sink(tmp_path)
    gym = _BarrierGym(sink, steps)
    bot = _FakeBot()
    model = _ScriptedModel([(step["steering"], step["event"]) for step in steps])

    monkeypatch.setattr(
        "farming.trainer.resolve_model_artifact", lambda path: Path("fake_model.zip")
    )
    monkeypatch.setattr(
        "farming.trainer.load_and_validate_model",
        lambda path, loader: SimpleNamespace(
            model=model,
            artifact_path=Path("fake_model.zip"),
            validation=SimpleNamespace(artifact_sha256="FAKESHA", contract_hash="FAKECONTRACT"),
        ),
    )
    monkeypatch.setattr(
        "farming.trainer.build_live_farming_runtime",
        lambda bot, config, token, status_callback=None: SimpleNamespace(
            gym=gym, close=lambda: None
        ),
    )

    # Let the poll thread sample at least one "no action observed yet"
    # frame before the control loop's first step.
    time.sleep(0.05)

    try:
        run_native_farming_agent(
            bot,
            config=FarmingRuntimeConfig(),
            cancellation=CancellationToken(),
            on_runtime_event=sink.add_runtime_event,
        )
        # A momentary EVA/jump step reverts to steering-only movement
        # right after gym.step() returns; give the poll thread a chance
        # to sample a frame in that post-revert window too before the
        # next scripted action overwrites it.
        time.sleep(0.05)
    finally:
        output_zip = sink.stop()

    archive = RecordingArchive(output_zip)
    frames_by_sequence = {frame.sequence: frame.action for frame in archive.frames()}
    max_sequence = max(frames_by_sequence)

    # 1. Initial no-action interval.
    assert frames_by_sequence[0] == -1

    scalar_sequence: list[int] = []
    for index, step in enumerate(steps):
        expected_during = _legacy_action(int(step["steering"]), int(step["event"]))
        sampled_range = gym.sampled_during[index]
        during_actions = {frames_by_sequence[seq] for seq in sampled_range}
        assert during_actions == {expected_during}, (
            f"step {index} ({step['label']}): frames {list(sampled_range)} "
            f"expected action {expected_during}, got {during_actions}"
        )
        scalar_sequence.append(expected_during)

        if index + 1 < len(steps):
            # The gap between this step's sampled frames and the next
            # step's sampled frames spans: this step's action_result
            # event, its revert event (if the step was momentary), the
            # next predict() call's simulated inference delay, and the
            # next step's action-publish event -- exactly the window
            # section 5/6 of the task require correct labeling for.
            gap = range(sampled_range.stop, gym.sampled_during[index + 1].start)
            assert len(gap) > 0, (
                f"no frame sampled between step {index} and step {index + 1}; "
                "the between-steps window was not observed"
            )
            gap_actions = {frames_by_sequence[seq] for seq in gap}
            if int(step["event"]) != int(FarmingEvent.NONE):
                # A momentary EVA/jump action must not stay stamped once
                # its step returns -- the between-steps frames must show
                # the still-held steering-only movement, never the
                # momentary action itself.
                expected_revert = _legacy_action(int(step["steering"]), int(FarmingEvent.NONE))
                assert gap_actions == {expected_revert}, (
                    f"step {index} ({step['label']}): between-steps frames "
                    f"{list(gap)} expected reverted action {expected_revert}, "
                    f"got {gap_actions}"
                )
            else:
                # Plain movement persists across the step boundary
                # unchanged -- no revert event is emitted for it.
                assert gap_actions == {expected_during}, (
                    f"step {index} ({step['label']}): between-steps frames "
                    f"{list(gap)} expected persisted action {expected_during}, "
                    f"got {gap_actions}"
                )

    # 6. episode_end must only clear the action after the final step's
    # frames were already visible with the final action -- the very last
    # sampled frame in the whole archive must be back at -1.
    assert frames_by_sequence[max_sequence] == -1

    # Codex's reproduced failure sequence resolved: every during-step
    # label matches the action actually issued for that step, in order.
    assert scalar_sequence == [1, 3, 4, 2, 0]

    events = {event.kind for event in archive.events()}
    assert {"action", "action_result", "episode_end"} <= events

    # 7. fit_world_model must see the correctly aligned action
    # distribution: every issued action (including the final one) must
    # be represented, with none of Codex's original shifted labels.
    map_data = MapModel.load()
    fitted = fit_world_model([output_zip], map_model=map_data)
    for expected in {0, 1, 2, 3, 4}:
        assert fitted.human_action_probabilities[expected] > 0, (
            f"legacy action {expected} is missing from the fitted "
            "world-model action distribution"
        )

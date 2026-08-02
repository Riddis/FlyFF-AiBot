from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from farming.config import FarmingRuntimeConfig
from farming.sb3_training import (
    SessionAwarePPO,
    TrainingBoundary,
    TrainingBoundaryKind,
)
from farming.trainer import (
    FarmingPreflight,
    FarmingRuntime,
    FarmingSessionServices,
    SessionStats,
    _TrainingCallback,
    train_native_farming,
)
from runtime_bus import RuntimeBus
from runtime_controller import RuntimeController
from worker_manager import CancellationToken


class FakeControl:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def release(self) -> None:
        self.events.append("release")


class FakeDomain:
    def __init__(self, events: list[str]) -> None:
        self.control = FakeControl(events)


class FakeRuntime:
    def __init__(self, events: list[str]) -> None:
        self.domain = FakeDomain(events)
        self.gym = object()
        self.preflight = FarmingPreflight(
            map_name="Tower AoE",
            map_hash="MAP-HASH",
            map_shape=(310, 294),
            player_base=0x1000,
            world_base=0x2000,
            pointer_generation=3,
            actor_cache_outcome="refreshed",
            actor_slots=64,
            initial_actor_count=7,
            initial_map_cell=(12, 34),
        )
        self.events = events

    def close(self) -> None:
        self.events.append("close")


class FakeBot:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.config: dict[str, object] = {}
        self.keyboard = None
        self.position_provider = None
        self.monster_provider = None
        self.native_process_service = None
        self.rl_enabled = False
        self.is_ready = True

    def start(self) -> None:
        self.events.append("start")
        self.rl_enabled = True

    def read_kill_count(self) -> int | None:
        return None

    def build_preview(self, frame: object) -> object:
        return frame

    def prepare_window(self, *_args: object) -> None:
        self.events.append("attach")

    def stop(self) -> None:
        self.events.append("stop")
        self.rl_enabled = False

    def stop_movement(self) -> None:
        self.events.append("stop_movement")

    def release_input(self) -> None:
        self.events.append("release_input")


class FakeModel:
    def __init__(
        self,
        events: list[str],
        *,
        failure: BaseException | None = None,
        boundary_after_learns: int = 1,
    ) -> None:
        self.events = events
        self.failure = failure
        self.boundary_after_learns = int(boundary_after_learns)
        self.learn_calls: list[dict[str, object]] = []
        self.num_timesteps = 17
        self.farming_contract_metadata: dict[str, object] | None = None
        self._boundary = TrainingBoundary(
            TrainingBoundaryKind.EXTERNAL_END,
            {
                "session_end_reason": "external_teleport",
                "session_classification": "external_truncation",
                "native_kill_delta": 2,
                "action_name": "RUN_FORWARD",
                "reward_components": {"kill": 2.0},
            },
            2.0,
        )
        self.session_boundary = self._boundary

    def learn(self, **kwargs: object) -> FakeModel:
        self.events.append("learn")
        self.learn_calls.append(dict(kwargs))
        if self.failure is not None:
            raise self.failure
        requested = int(kwargs.get("total_timesteps", 0))
        self.num_timesteps += requested
        if len(self.learn_calls) >= self.boundary_after_learns:
            self.session_boundary = self._boundary
        else:
            self.session_boundary = None
        return self

    def save(self, path: str) -> None:
        self.events.append("save")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("policy.txt", "fake-policy")


def _services(
    tmp_path: Path,
    runtime: FakeRuntime,
    model: FakeModel,
    events: list[str],
) -> FarmingSessionServices:
    def build_runtime(
        _bot: Any,
        _config: FarmingRuntimeConfig,
        _token: CancellationToken,
    ) -> FarmingRuntime:
        events.append("preflight")
        return cast(FarmingRuntime, cast(object, runtime))

    def load_model(
        _path: Path,
        _runtime: FarmingRuntime,
        _tensorboard: Path,
    ) -> tuple[SessionAwarePPO, bool]:
        events.append("model_preflight")
        return cast(SessionAwarePPO, cast(object, model)), False

    return FarmingSessionServices(
        runtime_builder=build_runtime,
        model_loader=load_model,
        path_resolver=lambda value: tmp_path / Path(value).name,
        session_path_factory=lambda _config, _kind: (
            tmp_path / "session.json",
            tmp_path / "session.manifest.json",
        ),
    )


def test_external_training_end_saves_model_report_and_recovery_manifest(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    runtime = FakeRuntime(events)
    model = FakeModel(events)
    bot = FakeBot(events)
    config = FarmingRuntimeConfig(checkpoint_frequency=8)

    result = train_native_farming(
        bot,  # pyright: ignore[reportArgumentType]
        config,
        cancellation=CancellationToken(),
        services=_services(tmp_path, runtime, model, events),
    )

    assert events[:4] == ["preflight", "model_preflight", "start", "learn"]
    assert events[-3:] == ["release", "save", "close"]
    assert result == tmp_path / "native_strategy_map_risk_ppo.zip"
    report = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert report["session_reason"] == "external_teleport"
    assert report["session_classification"] == "external_truncation"
    assert report["kills"] == 2
    assert report["total_reward"] == 2.0
    assert report["map"]["hash"] == "MAP-HASH"
    manifest = json.loads(
        (tmp_path / "session.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["path"] == str(result)
    assert len(manifest["model"]["sha256"]) == 64



def test_normal_training_status_is_concise_and_uses_total_model_steps() -> None:
    events: list[str] = []
    runtime = FakeRuntime(events)
    statuses: list[str] = []
    callback = _TrainingCallback(
        runtime=cast(FarmingRuntime, cast(object, runtime)),
        config=FarmingRuntimeConfig(stats_interval_seconds=0.001),
        cancellation=CancellationToken(),
        stats=SessionStats(),
        status_callback=statuses.append,
    )
    callback.model = SimpleNamespace(num_timesteps=123_456)
    callback.locals = {
        "infos": [
            {
                "native_kill_delta": 2,
                "action_name": "CAST_EVA",
                "jump_requested": False,
                "jump_performed": False,
            }
        ],
        "rewards": [1.75],
    }
    callback._last_status = 0.0

    assert callback._on_step()

    message = statuses[-1]
    assert message.startswith(
        "TRAINING | steps=123,456 reward=+1.75 reward_delta=+1.75 "
        "kills=2 kills/hr="
    )
    assert "action=CAST_EVA" in message
    for noisy_field in ("actors=", "cache=", "candidates=", "ocr="):
        assert noisy_field not in message


def test_training_continues_past_checkpoint_and_uses_total_model_steps(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    runtime = FakeRuntime(events)
    model = FakeModel(events, boundary_after_learns=2)
    bot = FakeBot(events)
    statuses: list[str] = []
    config = FarmingRuntimeConfig(checkpoint_frequency=8)

    result = train_native_farming(
        bot,  # pyright: ignore[reportArgumentType]
        config,
        status_callback=statuses.append,
        cancellation=CancellationToken(),
        services=_services(tmp_path, runtime, model, events),
    )

    assert result == tmp_path / "native_strategy_map_risk_ppo.zip"
    assert len(model.learn_calls) == 2
    assert model.learn_calls[0]["total_timesteps"] == 7
    assert model.learn_calls[0]["reset_num_timesteps"] is False
    assert model.learn_calls[1]["total_timesteps"] == 8
    assert model.learn_calls[1]["reset_num_timesteps"] is False
    assert model.num_timesteps == 32
    checkpoint = (
        tmp_path
        / "native_strategy_map_risk_checkpoints"
        / "native_strategy_map_risk_ppo_000000000024_steps.zip"
    )
    assert checkpoint.is_file()
    assert any(
        message.startswith("TRAINING STARTED | total_steps=17 checkpoint_every=8")
        for message in statuses
    )
    assert any(
        message.startswith("TRAINING CHECKPOINT | steps=24")
        for message in statuses
    )
    assert events.count("learn") == 2
    assert events.count("save") == 2


def test_fatal_training_failure_preserves_last_known_good_model(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    runtime = FakeRuntime(events)
    model = FakeModel(events, failure=RuntimeError("policy failure"))
    bot = FakeBot(events)
    destination = tmp_path / "native_strategy_map_risk_ppo.zip"
    destination.write_bytes(b"LAST-KNOWN-GOOD")

    try:
        train_native_farming(
            bot,  # pyright: ignore[reportArgumentType]
            FarmingRuntimeConfig(checkpoint_frequency=8),
            cancellation=CancellationToken(),
            services=_services(tmp_path, runtime, model, events),
        )
    except RuntimeError as error:
        assert str(error) == "policy failure"
    else:
        raise AssertionError("fatal training failure did not propagate")

    assert destination.read_bytes() == b"LAST-KNOWN-GOOD"
    assert "save" not in events
    assert events[-2:] == ["release", "close"]
    report = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert report["session_classification"] == "fatal_error"
    assert report["error"]["type"] == "RuntimeError"
    assert not (tmp_path / "session.manifest.json").exists()


def test_fake_launch_attach_preview_dry_run_and_external_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real_train = train_native_farming
    training_runtime = FakeRuntime(events)
    training_model = FakeModel(events)
    training_services = _services(
        tmp_path,
        training_runtime,
        training_model,
        events,
    )
    module = ModuleType("farming.trainer")

    def dry_run(selected_bot: FakeBot, **_kwargs: object) -> None:
        events.append("dry_preflight")
        selected_bot.start()
        events.append("dry_step")

    def train_mode(
        selected_bot: FakeBot,
        *,
        status_callback: Any,
        cancellation: CancellationToken,
    ) -> Path:
        return real_train(
            selected_bot,  # pyright: ignore[reportArgumentType]
            FarmingRuntimeConfig(checkpoint_frequency=8),
            status_callback=status_callback,
            cancellation=cancellation,
            services=training_services,
        )

    dynamic_module = cast(Any, module)
    dynamic_module.dry_run_native_farming = dry_run
    dynamic_module.train_native_farming = train_mode
    dynamic_module.run_native_farming_agent = lambda *_args, **_kwargs: None
    dynamic_module.validate_native_farming_data = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "farming.trainer", module)

    class Capture:
        def attach(self, _handle: int) -> int:
            events.append("capture")
            return 1

        def stop(self, _timeout: float) -> bool:
            events.append("capture_stop")
            return True

    class Preview:
        def start(self) -> None:
            events.append("preview")

        def stop(self, _timeout: float) -> bool:
            events.append("preview_stop")
            return True

    dry_bot = FakeBot(events)
    dry_controller = RuntimeController(
        dry_bot,  # pyright: ignore[reportArgumentType]
        RuntimeBus(),
    )
    dry_controller.capture = cast(Any, Capture())
    dry_controller.preview = cast(Any, Preview())
    def run_now(_name: str, target: Any) -> None:
        target(CancellationToken())

    cast(Any, dry_controller)._start_control = run_now
    dry_controller.attach(101)
    dry_controller.start_rl("dry-run")

    training_bot = FakeBot(events)
    training_controller = RuntimeController(
        training_bot,  # pyright: ignore[reportArgumentType]
        RuntimeBus(),
    )
    training_controller.capture = cast(Any, Capture())
    training_controller.preview = cast(Any, Preview())
    cast(Any, training_controller)._start_control = run_now
    training_controller.attach(202)
    training_controller.start_rl("train")

    assert events.count("capture") == 2
    assert events.count("preview") == 2
    assert events.index("dry_preflight") < events.index("dry_step")
    assert events.index("dry_step") < events.index("stop")
    assert events.index("preflight") < events.index("model_preflight")
    start_indexes = [index for index, event in enumerate(events) if event == "start"]
    assert len(start_indexes) == 2
    assert events.index("model_preflight") < start_indexes[-1]
    assert events.index("learn") < events.index("save")
    report = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert report["session_reason"] == "external_teleport"
    assert (tmp_path / "session.manifest.json").is_file()

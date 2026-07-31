from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from farming.reporting import atomic_save_model
from farming.sb3_adapter import (
    ExternalSessionEnded,
    FarmingSessionCancelled,
    PolicyTerminalDelivered,
)
from farming.sb3_training import (
    SessionAwarePPO,
    TerminalPrefixRolloutBuffer,
    TrainingBoundaryKind,
)
from farming.startup import load_and_validate_model


class BoundaryEnv(gym.Env[np.ndarray, int]):
    def __init__(self, boundary: TrainingBoundaryKind, *, prefix_steps: int) -> None:
        super().__init__()
        self.metadata: dict[str, Any] = {"render_modes": []}
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(482,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(4, start=0)
        self.boundary = boundary
        self.prefix_steps = prefix_steps
        self.steps = 0
        self.sink = False

    @staticmethod
    def _observation(value: float) -> np.ndarray:
        return np.full(482, value, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        if self.sink:
            return self._observation(float(self.steps)), {"terminal_sink": True}
        return self._observation(0.0), {}

    def _raised_step(self, classification: str) -> Any:
        return SimpleNamespace(
            info={"session_classification": classification},
            outcome=SimpleNamespace(reason=SimpleNamespace(value=classification)),
            reward=SimpleNamespace(total=0.0),
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        if self.sink:
            raise PolicyTerminalDelivered("terminal sink sampled")
        if self.steps >= self.prefix_steps:
            if self.boundary is TrainingBoundaryKind.EXTERNAL_END:
                raise ExternalSessionEnded(
                    self._raised_step("external_truncation")
                )
            if self.boundary is TrainingBoundaryKind.CANCELLED:
                raise FarmingSessionCancelled(
                    self._raised_step("user_cancellation")
                )
        self.steps += 1
        observation = self._observation(float(self.steps))
        if self.boundary is TrainingBoundaryKind.POLICY_TERMINAL:
            self.sink = True
            return (
                observation,
                -50.0,
                True,
                False,
                {"session_classification": "policy_termination"},
            )
        return observation, 1.0, False, False, {}


class TrainingSpyPPO(SessionAwarePPO):
    train_calls: int

    def __init__(self, env: gym.Env[np.ndarray, int]) -> None:
        super().__init__(
            "MlpPolicy",
            env,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            seed=7,
            verbose=0,
        )
        self.train_calls = 0

    def train(self) -> None:
        self.train_calls += 1
        super().train()


def test_policy_terminal_trains_only_the_real_prefix_once() -> None:
    environment = BoundaryEnv(TrainingBoundaryKind.POLICY_TERMINAL, prefix_steps=0)
    model = TrainingSpyPPO(environment)

    model.learn(total_timesteps=8)

    assert model.session_boundary is not None
    assert model.session_boundary.kind is TrainingBoundaryKind.POLICY_TERMINAL
    assert model.num_timesteps == 1
    assert model.train_calls == 1
    assert isinstance(model.rollout_buffer, TerminalPrefixRolloutBuffer)
    assert model.rollout_buffer.buffer_size == 1
    # The post-training sink attempt starts a new collection and resets the
    # buffer, but never reaches the environment as a real policy sample.
    assert model.rollout_buffer.pos == 0
    assert model.rollout_buffer.full is False
    assert environment.steps == 1


def test_external_end_discards_partial_rollout_without_training() -> None:
    environment = BoundaryEnv(TrainingBoundaryKind.EXTERNAL_END, prefix_steps=1)
    model = TrainingSpyPPO(environment)

    model.learn(total_timesteps=8)

    assert model.session_boundary is not None
    assert model.session_boundary.kind is TrainingBoundaryKind.EXTERNAL_END
    assert model.num_timesteps == 1
    assert model.train_calls == 0
    assert model.rollout_buffer.pos == 0
    assert model.rollout_buffer.full is False


def test_cancellation_is_not_counted_as_a_sample_or_training_boundary() -> None:
    environment = BoundaryEnv(TrainingBoundaryKind.CANCELLED, prefix_steps=0)
    model = TrainingSpyPPO(environment)

    model.learn(total_timesteps=8)

    assert model.session_boundary is not None
    assert model.session_boundary.kind is TrainingBoundaryKind.CANCELLED
    assert model.num_timesteps == 0
    assert model.train_calls == 0
    assert model.rollout_buffer.pos == 0
    assert model.rollout_buffer.full is False


def test_saved_policy_resumes_with_the_session_aware_buffer(
    tmp_path: Path,
) -> None:
    environment = BoundaryEnv(TrainingBoundaryKind.EXTERNAL_END, prefix_steps=1)
    model = TrainingSpyPPO(environment)
    artifact = tmp_path / "policy.zip"
    atomic_save_model(model, artifact)

    validated = load_and_validate_model(
        artifact,
        lambda value: SessionAwarePPO.load(
            value,
            custom_objects={
                "rollout_buffer_class": TerminalPrefixRolloutBuffer,
            },
        ),
    )

    resumed = validated.model
    assert isinstance(resumed, SessionAwarePPO)
    assert isinstance(resumed.rollout_buffer, TerminalPrefixRolloutBuffer)

from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep

from libs.HumanKeyboard import VKEY, HumanKeyboard, KeyPressTiming

from .RotationModel import (
    StateAwareRotationModel,
    TurnDirection,
    TurnPulseResult,
    TurnTransition,
    TurnTransitionTracker,
)


class MappingController:
    """Mapper-only AZERTY movement controller."""

    def __init__(
        self,
        keyboard: HumanKeyboard,
        *,
        neutral_after_seconds: float = 2.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.keyboard: HumanKeyboard = keyboard
        self.forward_key: int = VKEY["z"]
        self.left_key: int = VKEY["q"]
        self.right_key: int = VKEY["d"]
        self._clock: Callable[[], float] = clock
        self._turn_state: TurnTransitionTracker = TurnTransitionTracker(
            neutral_after_seconds
        )

    @property
    def neutral_after_seconds(self) -> float:
        return self._turn_state.neutral_after_seconds

    @property
    def previous_turn_direction(self) -> TurnDirection | None:
        return self._turn_state.last_direction

    @property
    def turn_idle_seconds(self) -> float | None:
        """Elapsed idle time since the last completed turn pulse."""
        return self._turn_state.idle_seconds(now=self._clock())

    def set_neutral_after_seconds(
        self,
        seconds: float,
        *,
        reset_history: bool = True,
    ) -> None:
        """Install a validated neutral timeout for subsequent turn decisions."""
        self._turn_state.set_neutral_after_seconds(seconds)
        if reset_history:
            self._turn_state.reset()

    def stop(self) -> None:
        """
        Use HumanKeyboard's public release API.

        The previous mapper called key_up() directly. That bypassed the normal
        input path used by the working RL actions and could leave Flyff treating
        a turn key as held after the short calibration pulse.
        """
        # Always send KEYUP for every mapper movement key. Relying only on
        # locally tracked keys is unsafe after cancellation or a failed
        # PostMessage call.
        first_error: Exception | None = None
        try:
            self.keyboard.release_keys(
                (
                    self.forward_key,
                    self.left_key,
                    self.right_key,
                )
            )
        except Exception as error:  # noqa: BLE001 - still release tracked keys.
            first_error = error
        try:
            self.keyboard.release_all()
        except Exception as error:  # noqa: BLE001 - report after both attempts.
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    def _pulse(self, key: int, seconds: float) -> KeyPressTiming:
        """
        Send one finite press through HumanKeyboard's tested public API.

        Do not compose a pulse from mapper-level key_down/sleep/key_up calls.
        HumanKeyboard owns the exact Windows message sequence required by the
        game.
        """
        duration = max(float(seconds), 0.015)
        self.stop()

        try:
            return self.keyboard.press_key(key, press_time=duration)
        finally:
            # HumanKeyboard also releases in a finally block. This redundant
            # mapper-level release covers alternate keyboard implementations
            # and ensures all movement keys are up between measurements.
            self.stop()

    def forward(self, seconds: float) -> KeyPressTiming:
        return self._pulse(self.forward_key, seconds)

    def turn_left(self, seconds: float) -> TurnPulseResult:
        return self._turn(TurnDirection.LEFT, self.left_key, seconds)

    def turn_right(self, seconds: float) -> TurnPulseResult:
        return self._turn(TurnDirection.RIGHT, self.right_key, seconds)

    def turn_degrees(
        self,
        direction: TurnDirection,
        degrees: float,
        rotation_model: StateAwareRotationModel,
        *,
        maximum_seconds: float,
    ) -> TurnPulseResult:
        """
        Classify state and choose its duration as one atomic controller action.

        This prevents a caller from predicting SAME/REVERSAL timing while the
        controller independently classifies the actual pulse as NEUTRAL.
        """
        self.stop()
        transition, idle_seconds = self._turn_state.classify(
            direction,
            now=self._clock(),
        )
        seconds = rotation_model.seconds_for(
            direction,
            transition,
            degrees,
            idle_seconds=idle_seconds,
        )
        seconds = min(maximum_seconds, max(0.015, seconds))
        key = self.left_key if direction is TurnDirection.LEFT else self.right_key
        return self._turn(
            direction,
            key,
            seconds,
            transition=transition,
            idle_seconds=idle_seconds,
            movement_released=True,
        )

    def _turn(
        self,
        direction: TurnDirection,
        key: int,
        seconds: float,
        *,
        transition: TurnTransition | None = None,
        idle_seconds: float | None = None,
        movement_released: bool = False,
    ) -> TurnPulseResult:
        requested = float(seconds)
        duration = max(requested, 0.015)
        if not movement_released:
            self.stop()
        if transition is None:
            transition, idle_seconds = self._turn_state.classify(
                direction,
                now=self._clock(),
            )
        try:
            # The idle timestamp above is sampled immediately before this
            # call, after all movement keys are confirmed released. It is the
            # closest controller-side measurement to target key-down.
            timing = self.keyboard.press_key(key, press_time=duration)
        finally:
            self.stop()
        completed_at = self._clock()
        self._turn_state.record(direction, completed_at=completed_at)
        return TurnPulseResult(
            direction=direction,
            transition=transition,
            requested_seconds=requested,
            clamped_seconds=timing.clamped_seconds,
            held_seconds=timing.held_seconds,
            elapsed_seconds=timing.elapsed_seconds,
            idle_seconds=idle_seconds,
        )

    def reset_turn_history(self) -> None:
        """Forget direction state after an intentionally long neutral wait."""
        self._turn_state.reset()

    def settle(self, seconds: float) -> None:
        self.stop()
        sleep(max(float(seconds), 0.0))

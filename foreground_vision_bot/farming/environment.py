from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import hypot, radians
from time import monotonic, sleep

import numpy as np
from numpy.typing import NDArray
from position.NativeFlyffMonsterProvider import ActorCacheOutcome

from .actions import FarmingAction, coerce_farming_action
from .config import FarmingRuntimeConfig
from .control import (
    DirectFarmingControl,
    FarmingControlCancelled,
    FarmingControlUnavailable,
)
from .kills import (
    CastWindow,
    NativeKillResult,
    NativeKillTracker,
    OcrDiagnostic,
    OcrKillDiagnostics,
)
from .map_context import FarmingMapContext
from .native_world import (
    NativeWorldFrame,
    NativeWorldReader,
    NativeWorldUnavailable,
    build_actor_observations,
)
from .observation import (
    BuiltObservation,
    ObservationBuilder,
    ObservationFrame,
    PlayerObservation,
)
from .reward import RewardCalculator, RewardConfig, RewardEvidence, RewardResult
from .session import (
    SessionEndReason,
    SessionEvidence,
    SessionOutcome,
    classify_session_outcome,
)

FloatArray = NDArray[np.float32]


class FarmingEnvironmentState(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    SEALED = "sealed"
    CLOSED = "closed"


class FarmingEnvironmentError(RuntimeError):
    pass


class FarmingEnvironmentUnavailable(FarmingEnvironmentError):
    pass


@dataclass(frozen=True, slots=True)
class FarmingReset:
    observation: FloatArray
    info: dict[str, object]


@dataclass(frozen=True, slots=True)
class FarmingStep:
    observation: FloatArray
    reward: RewardResult
    outcome: SessionOutcome
    info: dict[str, object]


@dataclass(frozen=True, slots=True)
class _FrameFeatures:
    world: NativeWorldFrame
    built: BuiltObservation
    player_cell: tuple[int, int]
    forbidden_distance_cells: float | None


@dataclass(frozen=True, slots=True)
class _TeleportConfirmation:
    suspected: bool = False
    confirmed: bool = False
    reason: str = "below_threshold"
    effective_threshold_cells: float = 0.0
    elapsed_seconds: float = 0.0
    thresholds_cells: tuple[float, ...] = ()
    selected_sample_index: int = 0
    displacements_cells: tuple[float, ...] = ()
    destination_spread_cells: float | None = None
    player_identity_stable: bool = True
    player_identity_changed: bool = False
    usable_sample_found: bool = True
    before_player_base: int = 0
    before_pointer_slot: int = 0
    before_native_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    before_pose_timestamp: float = 0.0
    player_bases: tuple[int, ...] = ()
    pointer_slots: tuple[int, ...] = ()
    native_positions: tuple[tuple[float, float, float], ...] = ()
    pose_timestamps: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class _UnexpectedTeleportResponse:
    source_in_mapped_teleport_area: bool = False
    pulse_attempted: bool = False
    pulse_completed: bool = False
    pulse_seconds: float = 0.0
    pulse_error: str | None = None


class UnifiedFarmingEnv:
    """One explicit live farming session with exactly one reset."""

    def __init__(
        self,
        world_reader: NativeWorldReader,
        map_context: FarmingMapContext,
        control: DirectFarmingControl,
        cancellation: object,
        *,
        config: FarmingRuntimeConfig | None = None,
        observation_builder: ObservationBuilder | None = None,
        reward_calculator: RewardCalculator | None = None,
        kill_tracker: NativeKillTracker | None = None,
        ocr_diagnostics: OcrKillDiagnostics | None = None,
        read_ocr_kills: Callable[[], int | None] | None = None,
        diagnostic_sink: Callable[[Mapping[str, object]], None] | None = None,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if control.cancellation is not cancellation:
            raise ValueError(
                "Environment and direct control must share the identical worker token"
            )
        self.world_reader = world_reader
        self.map_context = map_context
        self.control = control
        self.cancellation = cancellation
        self.config = config or FarmingRuntimeConfig()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.reward_calculator = reward_calculator or RewardCalculator(
            RewardConfig(
                teleport_warning_radius_cells=(
                    self.config.teleport_warning_radius_cells
                ),
                teleport_buffer_radius_cells=self.config.teleport_buffer_radius_cells,
                teleport_proximity_penalty=self.config.teleport_proximity_penalty,
                teleport_buffer_penalty=self.config.teleport_buffer_penalty,
                teleport_trigger_penalty=self.config.teleport_trigger_penalty,
                jump_flair_reward=self.config.jump_flair_reward,
            )
        )
        self.kill_tracker = kill_tracker or NativeKillTracker(
            zero_hp_confirmation_reads=self.config.kill_zero_confirmation_reads,
            result_timeout_seconds=self.config.cast_result_timeout_seconds,
            poll_seconds=self.config.cast_poll_seconds,
            clock=clock,
            sleeper=sleeper,
        )
        self.ocr_diagnostics = ocr_diagnostics or OcrKillDiagnostics()
        self._read_ocr_kills = read_ocr_kills
        self._diagnostic_sink = diagnostic_sink
        self._clock = clock
        self._sleep = sleeper
        self._state = FarmingEnvironmentState.NEW
        self._started_at: float | None = None
        self._last_cast_at: float | None = None
        self._last_jump_at: float | None = None
        self._last_features: _FrameFeatures | None = None
        self._terminal_observation: FloatArray | None = None

    @property
    def state(self) -> FarmingEnvironmentState:
        return self._state

    @property
    def terminal_observation(self) -> FloatArray | None:
        if self._terminal_observation is None:
            return None
        result = self._terminal_observation.copy()
        result.setflags(write=False)
        return result

    @staticmethod
    def _cancelled(cancellation: object) -> bool:
        cancelled = getattr(cancellation, "cancelled", None)
        if cancelled is not None:
            return bool(cancelled() if callable(cancelled) else cancelled)
        is_set = getattr(cancellation, "is_set", None)
        return bool(is_set()) if callable(is_set) else False

    def _wait(self, seconds: float) -> None:
        wait = getattr(self.cancellation, "wait", None)
        if callable(wait):
            wait(max(0.0, seconds))
        else:
            self._sleep(max(0.0, seconds))

    def _ensure_state(self, expected: FarmingEnvironmentState) -> None:
        if self._state is not expected:
            raise FarmingEnvironmentError(
                f"Farming environment is {self._state.value}; expected {expected.value}"
            )

    def _cooldown_fraction(self, now: float) -> float:
        if self._last_cast_at is None:
            return 1.0
        return float(
            np.clip(
                (now - self._last_cast_at) / self.config.eva_cooldown_seconds,
                0.0,
                1.0,
            )
        )

    def _jump_cooldown_fraction(self, now: float) -> float:
        if self._last_jump_at is None:
            return 1.0
        return float(
            np.clip(
                (now - self._last_jump_at) / self.config.jump_cooldown_seconds,
                0.0,
                1.0,
            )
        )

    def _jump_available(self, before: _FrameFeatures, now: float) -> bool:
        # The policy may request forward+jump from any movement state.  The
        # action itself is never masked or degraded; selecting it always holds
        # forward and taps Space.  ``jump_cooldown_seconds`` limits only how
        # often the tiny flair reward can be earned, preventing reward farming
        # without restricting the physical action.
        del before, now
        return True

    def _jump_reward_available(self, now: float) -> bool:
        return self._jump_cooldown_fraction(now) >= 1.0

    def _build_features(
        self,
        world: NativeWorldFrame,
        *,
        action: FarmingAction,
        displacement_cells: float,
        contact: bool,
        now: float,
    ) -> _FrameFeatures:
        player_cell = self.map_context.native_to_layout_cell(
            world.player_pose.x,
            world.player_pose.z,
        )
        if player_cell is None:
            raise FarmingEnvironmentUnavailable(
                "Native player position is outside the selected farming map"
            )
        normalized_x, normalized_z = self.map_context.features.normalized_position(
            player_cell
        )
        actor_observations = build_actor_observations(
            world,
            self.map_context,
            maximum_geodesic_distance_cells=self.config.vision_radius_cells,
        )
        cooldown = self._cooldown_fraction(now)
        heading = world.player_pose.heading_degrees
        player = PlayerObservation(
            normalized_x=normalized_x,
            normalized_z=normalized_z,
            heading_radians=0.0 if heading is None else radians(heading),
            eva_cooldown_fraction=cooldown,
            displacement_cells=displacement_cells,
            contact=contact,
            held_movement=self.control.held_movement,
            last_policy_action=action,
            jump_cooldown_fraction=self._jump_cooldown_fraction(now),
            map_available=True,
        )
        built = self.observation_builder.build(
            ObservationFrame(
                player=player,
                actors=actor_observations,
                local_map=self.map_context.features.local_crop(player_cell),
            )
        )
        return _FrameFeatures(
            world=world,
            built=built,
            player_cell=player_cell,
            forbidden_distance_cells=self.map_context.features.forbidden_distance(
                player_cell
            ),
        )

    @staticmethod
    def _actor_payload(actor: object) -> dict[str, object]:
        return {
            "base": int(getattr(actor, "base_address")),
            "species": int(getattr(actor, "species_id")),
            "active_species": int(getattr(actor, "active_species_id")),
            "hp": int(getattr(actor, "hp")),
            "x": float(getattr(actor, "x")),
            "y": float(getattr(actor, "y")),
            "z": float(getattr(actor, "z")),
            "distance_native": float(getattr(actor, "distance_native")),
            "hp_offset": (
                None
                if getattr(actor, "hp_offset", None) is None
                else int(getattr(actor, "hp_offset"))
            ),
            "hp_candidates": [
                {"offset": int(offset), "value": int(value)}
                for offset, value in tuple(getattr(actor, "hp_candidates", ()))
            ],
            "hp_offset_validated": bool(
                getattr(actor, "hp_offset_validated", False)
            ),
        }

    @classmethod
    def _world_payload(cls, world: NativeWorldFrame) -> dict[str, object]:
        snapshot = world.pointer_snapshot
        pose = world.player_pose
        return {
            "pointer": {
                "player_base": int(snapshot.player_base),
                "world_base": int(snapshot.world_base),
                "generation": int(snapshot.generation),
                "mode": str(snapshot.mode),
            },
            "player": {
                "x": float(pose.x),
                "y": float(pose.y),
                "z": float(pose.z),
                "heading_degrees": (
                    None
                    if pose.heading_degrees is None
                    else float(pose.heading_degrees)
                ),
            },
            "actors": [cls._actor_payload(actor) for actor in world.actors],
            "tracked_actors": [
                cls._actor_payload(actor) for actor in world.tracked_actors
            ],
        }

    def _emit_diagnostic(
        self,
        *,
        event: str,
        action: FarmingAction,
        before: _FrameFeatures | None,
        after: _FrameFeatures,
        observation: FloatArray,
        info: Mapping[str, object],
        eva_available: bool | None = None,
        cast_window: CastWindow | None = None,
        kill_result: NativeKillResult | None = None,
        ocr: OcrDiagnostic | None = None,
    ) -> None:
        sink = self._diagnostic_sink
        if sink is None:
            return
        vector = np.asarray(observation, dtype=np.float32)
        finite = np.isfinite(vector)
        finite_values = vector[finite]
        payload: dict[str, object] = {
            "event": str(event),
            "timestamp_monotonic": float(self._clock()),
            "action": int(action),
            "action_name": action.name,
            "eva_available": eva_available,
            "before": None if before is None else self._world_payload(before.world),
            "after": self._world_payload(after.world),
            "observation": {
                "size": int(vector.size),
                "finite": bool(finite.all()),
                "nonfinite_count": int(vector.size - int(finite.sum())),
                "minimum": (
                    float(finite_values.min()) if finite_values.size else None
                ),
                "maximum": (
                    float(finite_values.max()) if finite_values.size else None
                ),
                "values": [float(value) for value in vector.tolist()],
            },
            "info": dict(info),
            "cast_candidates": (
                []
                if cast_window is None
                else [
                    {
                        "base": int(candidate.base_address),
                        "species": int(candidate.species_id),
                        "initial_hp": int(candidate.initial_hp),
                    }
                    for candidate in cast_window.candidates
                ]
            ),
            "kill_result": (
                None
                if kill_result is None
                else {
                    "confirmed": [
                        {
                            "base": int(candidate.base_address),
                            "species": int(candidate.species_id),
                        }
                        for candidate in kill_result.confirmed
                    ],
                    "polls": int(kill_result.polls),
                    "successful_reads": int(kill_result.successful_reads),
                    "failed_reads": int(kill_result.failed_reads),
                    "cancelled": bool(kill_result.cancelled),
                    "elapsed_seconds": float(kill_result.elapsed_seconds),
                    "candidate_diagnostics": [
                        {
                            "base": int(item.base_address),
                            "species": int(item.species_id),
                            "present_reads": int(item.present_reads),
                            "absent_reads": int(item.absent_reads),
                            "maximum_consecutive_absence": int(
                                item.maximum_consecutive_absence
                            ),
                            "minimum_seen_hp": item.minimum_seen_hp,
                            "last_seen_hp": item.last_seen_hp,
                            "initial_hp": item.initial_hp,
                            "zero_hp_reads": int(item.zero_hp_reads),
                            "maximum_consecutive_zero_hp": int(
                                item.maximum_consecutive_zero_hp
                            ),
                            "hp_decreased": bool(item.hp_decreased),
                            "confirmed": bool(item.confirmed),
                        }
                        for item in kill_result.diagnostics
                    ],
                }
            ),
            "ocr": (
                None
                if ocr is None
                else {
                    "outcome": ocr.outcome.value,
                    "value": ocr.value,
                    "previous": ocr.previous,
                    "delta": ocr.delta,
                }
            ),
        }
        sink(payload)

    def reset(self) -> FarmingReset:
        self._ensure_state(FarmingEnvironmentState.NEW)
        if self._cancelled(self.cancellation):
            self._seal()
            raise FarmingControlCancelled("Farming reset was cancelled")
        world = self.world_reader.read_frame()
        now = self._clock()
        features = self._build_features(
            world,
            action=FarmingAction.RUN_FORWARD,
            displacement_cells=0.0,
            contact=False,
            now=now,
        )
        if self.map_context.features.is_forbidden(features.player_cell):
            self._seal(features.built.vector)
            raise FarmingEnvironmentUnavailable(
                "Player starts inside the mapped teleport trigger"
            )
        self._started_at = now
        self._last_features = features
        self._state = FarmingEnvironmentState.ACTIVE
        info = self._info(
            action=FarmingAction.RUN_FORWARD,
            features=features,
            reward=None,
            outcome=SessionOutcome.continuing(),
            native_kills=0,
            ocr=None,
            eva_available=True,
        )
        self._emit_diagnostic(
            event="reset",
            action=FarmingAction.RUN_FORWARD,
            before=None,
            after=features,
            observation=features.built.vector,
            info=info,
            eva_available=True,
        )
        return FarmingReset(observation=features.built.vector, info=info)

    def _read_after_with_grace(self) -> NativeWorldFrame:
        deadline = self._clock() + self.config.pointer_grace_seconds
        last_error: Exception | None = None
        while self._clock() < deadline:
            if self._cancelled(self.cancellation):
                raise FarmingControlCancelled("Farming step was cancelled")
            try:
                return self.world_reader.read_frame()
            except NativeWorldUnavailable as error:
                if error.outcome is ActorCacheOutcome.WORLD_MISMATCH:
                    raise
                last_error = error
                self._wait(self.config.pointer_poll_seconds)
        raise NativeWorldUnavailable(
            ActorCacheOutcome.UNAVAILABLE,
            f"Native world remained unavailable through pointer grace: {last_error}",
        )

    def _displacement_cells(
        self,
        before: NativeWorldFrame,
        after: NativeWorldFrame,
    ) -> float:
        return float(
            hypot(
                after.player_pose.x - before.player_pose.x,
                after.player_pose.z - before.player_pose.z,
            )
            / self.map_context.native_units_per_cell
        )

    @staticmethod
    def _frame_identity(frame: NativeWorldFrame) -> tuple[int, int]:
        snapshot = frame.pointer_snapshot
        return int(snapshot.player_base), int(snapshot.player_pointer_address)

    @staticmethod
    def _native_position(frame: NativeWorldFrame) -> tuple[float, float, float]:
        pose = frame.player_pose
        return float(pose.x), float(pose.y), float(pose.z)

    @staticmethod
    def _elapsed_pose_seconds(
        before: NativeWorldFrame,
        after: NativeWorldFrame,
    ) -> float:
        return max(
            0.0,
            float(after.player_pose.timestamp) - float(before.player_pose.timestamp),
        )

    def _teleport_threshold_cells(
        self,
        before: NativeWorldFrame,
        after: NativeWorldFrame,
    ) -> float:
        elapsed = self._elapsed_pose_seconds(before, after)
        motion_allowance = (
            self.config.teleport_motion_allowance_cells_per_second * elapsed
            + self.config.teleport_motion_margin_cells
        )
        return float(
            max(self.config.teleport_jump_threshold_cells, motion_allowance)
        )

    def _destination_spread_cells(
        self,
        samples: tuple[NativeWorldFrame, ...],
    ) -> float:
        maximum = 0.0
        units = self.map_context.native_units_per_cell
        for index, left in enumerate(samples):
            for right in samples[index + 1 :]:
                maximum = max(
                    maximum,
                    hypot(
                        right.player_pose.x - left.player_pose.x,
                        right.player_pose.z - left.player_pose.z,
                    )
                    / units,
                )
        return float(maximum)

    def _confirm_teleport_sample(
        self,
        before: NativeWorldFrame,
        first_after: NativeWorldFrame,
    ) -> tuple[NativeWorldFrame, _TeleportConfirmation]:
        first_displacement = self._displacement_cells(before, first_after)
        first_threshold = self._teleport_threshold_cells(before, first_after)
        first_elapsed = self._elapsed_pose_seconds(before, first_after)
        if first_displacement < first_threshold:
            return first_after, _TeleportConfirmation(
                reason="below_threshold",
                effective_threshold_cells=first_threshold,
                elapsed_seconds=first_elapsed,
                thresholds_cells=(first_threshold,),
                displacements_cells=(first_displacement,),
                player_identity_stable=True,
                player_identity_changed=(
                    self._frame_identity(first_after) != self._frame_identity(before)
                ),
                usable_sample_found=True,
                before_player_base=int(before.pointer_snapshot.player_base),
                before_pointer_slot=int(
                    before.pointer_snapshot.player_pointer_address
                ),
                before_native_position=self._native_position(before),
                before_pose_timestamp=float(before.player_pose.timestamp),
                player_bases=(int(first_after.pointer_snapshot.player_base),),
                pointer_slots=(
                    int(first_after.pointer_snapshot.player_pointer_address),
                ),
                native_positions=(self._native_position(first_after),),
                pose_timestamps=(float(first_after.player_pose.timestamp),),
            )

        samples: list[NativeWorldFrame] = [first_after]
        while len(samples) < self.config.teleport_confirmation_samples:
            self._wait(self.config.teleport_confirmation_interval_seconds)
            samples.append(self._read_after_with_grace())

        sample_tuple = tuple(samples)
        displacements = tuple(
            self._displacement_cells(before, sample) for sample in sample_tuple
        )
        thresholds = tuple(
            self._teleport_threshold_cells(before, sample) for sample in sample_tuple
        )
        identities = tuple(self._frame_identity(sample) for sample in sample_tuple)
        before_identity = self._frame_identity(before)
        same_as_before = tuple(
            identity == before_identity for identity in identities
        )
        after_identity_stable = len(set(identities)) == 1
        after_identity_changed = bool(
            after_identity_stable and identities[0] != before_identity
        )
        spread = self._destination_spread_cells(sample_tuple)
        all_far = all(
            displacement >= threshold
            for displacement, threshold in zip(
                displacements,
                thresholds,
                strict=True,
            )
        )
        destination_stable = (
            spread <= self.config.teleport_confirmation_tolerance_cells
        )
        confirmed = bool(
            after_identity_stable and all_far and destination_stable
        )

        if confirmed:
            selected_index = len(sample_tuple) - 1
            reason = (
                "stable_repeated_discontinuity_new_player"
                if after_identity_changed
                else "stable_repeated_discontinuity_same_player"
            )
        else:
            safe_matching_indices = [
                index
                for index, matches in enumerate(same_as_before)
                if matches and displacements[index] < thresholds[index]
            ]
            matching_indices = [
                index for index, matches in enumerate(same_as_before) if matches
            ]
            if safe_matching_indices:
                selected_index = safe_matching_indices[-1]
            else:
                pool = matching_indices or list(range(len(sample_tuple)))
                selected_index = min(pool, key=lambda index: displacements[index])
            if not after_identity_stable:
                reason = "player_identity_unstable"
            elif not all_far:
                reason = "returned_below_threshold"
            else:
                reason = "destination_unstable"

        selected = sample_tuple[selected_index]
        return selected, _TeleportConfirmation(
            suspected=True,
            confirmed=confirmed,
            reason=reason,
            effective_threshold_cells=max(thresholds),
            elapsed_seconds=max(
                self._elapsed_pose_seconds(before, sample) for sample in sample_tuple
            ),
            thresholds_cells=thresholds,
            selected_sample_index=selected_index,
            displacements_cells=displacements,
            destination_spread_cells=spread,
            player_identity_stable=after_identity_stable,
            player_identity_changed=after_identity_changed,
            usable_sample_found=bool(confirmed or safe_matching_indices),
            before_player_base=int(before.pointer_snapshot.player_base),
            before_pointer_slot=int(
                before.pointer_snapshot.player_pointer_address
            ),
            before_native_position=self._native_position(before),
            before_pose_timestamp=float(before.player_pose.timestamp),
            player_bases=tuple(identity[0] for identity in identities),
            pointer_slots=tuple(identity[1] for identity in identities),
            native_positions=tuple(
                self._native_position(sample) for sample in sample_tuple
            ),
            pose_timestamps=tuple(
                float(sample.player_pose.timestamp) for sample in sample_tuple
            ),
        )

    def _respond_to_unexpected_teleport(
        self,
        *,
        source_in_mapped_teleport_area: bool,
    ) -> _UnexpectedTeleportResponse:
        if source_in_mapped_teleport_area:
            return _UnexpectedTeleportResponse(
                source_in_mapped_teleport_area=True,
            )

        seconds = float(self.config.unexpected_teleport_forward_pulse_seconds)
        try:
            self.control.pulse_forward(seconds)
        except Exception as error:  # Stop even when the recovery pulse fails.
            try:
                self.control.release()
            except Exception:
                pass
            return _UnexpectedTeleportResponse(
                source_in_mapped_teleport_area=False,
                pulse_attempted=True,
                pulse_completed=False,
                pulse_seconds=seconds,
                pulse_error=f"{type(error).__name__}: {error}",
            )
        return _UnexpectedTeleportResponse(
            source_in_mapped_teleport_area=False,
            pulse_attempted=True,
            pulse_completed=True,
            pulse_seconds=seconds,
        )

    def step(self, action: FarmingAction | int) -> FarmingStep:
        self._ensure_state(FarmingEnvironmentState.ACTIVE)
        selected = coerce_farming_action(action)
        before = self._last_features
        assert before is not None
        started = self._clock()
        eva_available = self._cooldown_fraction(started) >= 1.0
        jump_requested = selected is FarmingAction.RUN_FORWARD_JUMP
        jump_available = bool(
            jump_requested and self._jump_available(before, started)
        )
        jump_reward_available = bool(
            jump_requested and self._jump_reward_available(started)
        )
        executed = selected
        jump_performed = bool(jump_requested)
        kill_result = NativeKillResult((), 0, 0, 0, False, 0.0)
        ocr: OcrDiagnostic | None = None
        try:
            cast_window = None
            if selected is FarmingAction.CAST_EVA and eva_available:
                cast_window = self.kill_tracker.begin_cast(before.world)
            self.control.execute(executed)
            if jump_reward_available:
                self._last_jump_at = started
            if selected is FarmingAction.CAST_EVA:
                self._last_cast_at = started
                if cast_window is not None:
                    kill_result = self.kill_tracker.confirm_cast(
                        cast_window,
                        read_actor_hp_states=(
                            self.world_reader.read_actor_hp_states
                        ),
                        cancellation=self.cancellation,
                    )
            else:
                self._wait(self.config.control_interval_seconds)

            if self._read_ocr_kills is not None:
                ocr = self.ocr_diagnostics.observe(self._read_ocr_kills())
            first_after_world = self._read_after_with_grace()
            after_world, teleport = self._confirm_teleport_sample(
                before.world,
                first_after_world,
            )
            finished = self._clock()
            displacement = self._displacement_cells(before.world, after_world)
            source_in_mapped_teleport_area = bool(
                self.map_context.features.is_forbidden(before.player_cell)
                or (
                    before.forbidden_distance_cells is not None
                    and before.forbidden_distance_cells
                    <= self.config.teleport_buffer_radius_cells
                )
            )
            if teleport.confirmed:
                response = self._respond_to_unexpected_teleport(
                    source_in_mapped_teleport_area=source_in_mapped_teleport_area,
                )
                if source_in_mapped_teleport_area:
                    outcome = SessionOutcome.forbidden_zone_entered()
                    event_name = "expected_map_teleport"
                else:
                    outcome = classify_session_outcome(
                        SessionEvidence(
                            user_cancelled=self._cancelled(self.cancellation),
                            session_time_expired=bool(
                                self._started_at is not None
                                and finished - self._started_at
                                >= self.config.episode_seconds
                            ),
                            map_transition=(
                                after_world.world_base != before.world.world_base
                            ),
                            external_teleport_confirmed=True,
                            displacement_cells=displacement,
                            teleport_jump_threshold_cells=(
                                self.config.teleport_jump_threshold_cells
                            ),
                        )
                    )
                    event_name = "external_teleport"
                reward = self.reward_calculator.calculate(
                    RewardEvidence(
                        native_kill_delta=kill_result.kill_count,
                        forbidden_distance_cells=(
                            0.0 if source_in_mapped_teleport_area else None
                        ),
                        session_outcome=outcome,
                    )
                )
                info = self._info(
                    action=selected,
                    features=before,
                    reward=reward,
                    outcome=outcome,
                    native_kills=kill_result.kill_count,
                    ocr=ocr,
                    kill_result=kill_result,
                    cast_window=cast_window,
                    eva_available=eva_available,
                    displacement_cells=displacement,
                    teleport=teleport,
                    teleport_response=response,
                    executed_action=executed,
                    jump_requested=jump_requested,
                    jump_available=jump_available,
                    jump_reward_available=jump_reward_available,
                    jump_performed=jump_performed,
                )
                self._emit_diagnostic(
                    event=event_name,
                    action=selected,
                    before=before,
                    after=before,
                    observation=before.built.vector,
                    info=info,
                    eva_available=eva_available,
                    cast_window=cast_window,
                    kill_result=kill_result,
                    ocr=ocr,
                )
                self._seal(before.built.vector)
                return FarmingStep(before.built.vector, reward, outcome, info)

            if teleport.suspected and not teleport.usable_sample_found:
                outcome = SessionOutcome.external(
                    SessionEndReason.POINTER_GRACE_EXHAUSTED,
                    "coordinate confirmation samples remained incoherent; "
                    "training stopped before consuming an untrusted position",
                )
                reward = self.reward_calculator.calculate(
                    RewardEvidence(
                        native_kill_delta=kill_result.kill_count,
                        session_outcome=outcome,
                    )
                )
                info = self._info(
                    action=selected,
                    features=before,
                    reward=reward,
                    outcome=outcome,
                    native_kills=kill_result.kill_count,
                    ocr=ocr,
                    kill_result=kill_result,
                    cast_window=cast_window,
                    eva_available=eva_available,
                    displacement_cells=displacement,
                    teleport=teleport,
                    executed_action=executed,
                    jump_requested=jump_requested,
                    jump_available=jump_available,
                    jump_reward_available=jump_reward_available,
                    jump_performed=jump_performed,
                )
                self._emit_diagnostic(
                    event="incoherent_position_samples",
                    action=selected,
                    before=before,
                    after=before,
                    observation=before.built.vector,
                    info=info,
                    eva_available=eva_available,
                    cast_window=cast_window,
                    kill_result=kill_result,
                    ocr=ocr,
                )
                self._seal(before.built.vector)
                return FarmingStep(before.built.vector, reward, outcome, info)

            contact = bool(selected.is_movement and displacement <= 0.05)
            after = self._build_features(
                after_world,
                action=selected,
                displacement_cells=displacement,
                contact=contact,
                now=finished,
            )
            session_evidence = SessionEvidence(
                user_cancelled=self._cancelled(self.cancellation),
                session_time_expired=bool(
                    self._started_at is not None
                    and finished - self._started_at >= self.config.episode_seconds
                ),
                map_transition=(after.world.world_base != before.world.world_base),
                external_teleport_confirmed=teleport.confirmed,
                sampled_forbidden_occupancy=self.map_context.features.is_forbidden(
                    after.player_cell
                ),
                started_inside_warning_radius=bool(
                    before.forbidden_distance_cells is not None
                    and before.forbidden_distance_cells
                    < self.config.teleport_warning_radius_cells
                ),
                displacement_cells=displacement,
                teleport_jump_threshold_cells=(
                    self.config.teleport_jump_threshold_cells
                ),
            )
            outcome = classify_session_outcome(session_evidence)
            density_delta = after.built.eva_actor_count - before.built.eva_actor_count
            reward = self.reward_calculator.calculate(
                RewardEvidence(
                    native_kill_delta=kill_result.kill_count,
                    density_delta=density_delta,
                    elapsed_seconds=max(0.0, finished - started),
                    eva_attempted=selected is FarmingAction.CAST_EVA,
                    eva_available=eva_available,
                    contact=contact,
                    jump_performed=jump_reward_available,
                    forbidden_distance_cells=after.forbidden_distance_cells,
                    session_outcome=outcome,
                )
            )
            info = self._info(
                action=selected,
                features=after,
                reward=reward,
                outcome=outcome,
                native_kills=kill_result.kill_count,
                ocr=ocr,
                kill_result=kill_result,
                cast_window=cast_window,
                eva_available=eva_available,
                displacement_cells=displacement,
                contact=contact,
                teleport=teleport,
                executed_action=executed,
                jump_requested=jump_requested,
                jump_available=jump_available,
                jump_reward_available=jump_reward_available,
                jump_performed=jump_performed,
            )
            self._emit_diagnostic(
                event="step",
                action=selected,
                before=before,
                after=after,
                observation=after.built.vector,
                info=info,
                eva_available=eva_available,
                cast_window=cast_window,
                kill_result=kill_result,
                ocr=ocr,
            )
            self._last_features = after
            if outcome.should_stop_session:
                self._seal(after.built.vector)
            return FarmingStep(after.built.vector, reward, outcome, info)
        except FarmingControlCancelled:
            outcome = SessionOutcome.cancelled("worker cancellation was requested")
            reward = self.reward_calculator.calculate(
                RewardEvidence(session_outcome=outcome)
            )
            self._seal(before.built.vector)
            return FarmingStep(
                before.built.vector,
                reward,
                outcome,
                self._info(
                    action=selected,
                    features=before,
                    reward=reward,
                    outcome=outcome,
                    native_kills=0,
                    ocr=ocr,
                ),
            )
        except FarmingControlUnavailable as error:
            outcome = classify_session_outcome(SessionEvidence(focus_lost=True))
            reward = self.reward_calculator.calculate(
                RewardEvidence(session_outcome=outcome)
            )
            self._seal(before.built.vector)
            return FarmingStep(
                before.built.vector,
                reward,
                outcome,
                self._info(
                    action=selected,
                    features=before,
                    reward=reward,
                    outcome=outcome,
                    native_kills=0,
                    ocr=ocr,
                    detail=str(error),
                ),
            )
        except NativeWorldUnavailable as error:
            outcome = classify_session_outcome(
                SessionEvidence(
                    map_transition=error.outcome is ActorCacheOutcome.WORLD_MISMATCH,
                    pointer_grace_exhausted=error.outcome
                    is not ActorCacheOutcome.WORLD_MISMATCH,
                )
            )
            reward = self.reward_calculator.calculate(
                RewardEvidence(session_outcome=outcome)
            )
            self._seal(before.built.vector)
            return FarmingStep(
                before.built.vector,
                reward,
                outcome,
                self._info(
                    action=selected,
                    features=before,
                    reward=reward,
                    outcome=outcome,
                    native_kills=0,
                    ocr=ocr,
                    detail=str(error),
                ),
            )
        except Exception:
            self._seal(before.built.vector)
            raise

    def _info(
        self,
        *,
        action: FarmingAction,
        features: _FrameFeatures,
        reward: RewardResult | None,
        outcome: SessionOutcome,
        native_kills: int,
        ocr: OcrDiagnostic | None,
        detail: str = "",
        kill_result: NativeKillResult | None = None,
        cast_window: CastWindow | None = None,
        eva_available: bool | None = None,
        displacement_cells: float = 0.0,
        contact: bool = False,
        teleport: _TeleportConfirmation | None = None,
        teleport_response: _UnexpectedTeleportResponse | None = None,
        executed_action: FarmingAction | None = None,
        jump_requested: bool = False,
        jump_available: bool = False,
        jump_reward_available: bool = False,
        jump_performed: bool = False,
    ) -> dict[str, object]:
        tracked_actors = (
            features.world.tracked_actors
            if features.world.tracked_actors
            else features.world.actors
        )
        hp_actor = tracked_actors[0] if tracked_actors else None
        actor_reader = getattr(self.world_reader, "actor_reader", None)
        actor_diagnostics = getattr(actor_reader, "last_diagnostics", None)
        return {
            "action": int(action),
            "action_name": action.name,
            "native_kill_delta": int(native_kills),
            "kill_delta": int(native_kills),
            "native_hp_offset": (
                None
                if hp_actor is None or getattr(hp_actor, "hp_offset", None) is None
                else int(getattr(hp_actor, "hp_offset"))
            ),
            "native_hp_candidate_offsets": (
                []
                if hp_actor is None
                else [
                    int(offset)
                    for offset, _value in tuple(
                        getattr(hp_actor, "hp_candidates", ())
                    )
                ]
            ),
            "native_hp_offset_validated": bool(
                False
                if hp_actor is None
                else getattr(hp_actor, "hp_offset_validated", False)
            ),
            "native_kill_candidates": (
                0 if cast_window is None else len(cast_window.candidates)
            ),
            "native_kill_confirmed": (
                []
                if kill_result is None
                else [int(item.base_address) for item in kill_result.confirmed]
            ),
            "native_kill_polls": 0 if kill_result is None else kill_result.polls,
            "native_kill_successful_reads": (
                0 if kill_result is None else kill_result.successful_reads
            ),
            "native_kill_failed_reads": (
                0 if kill_result is None else kill_result.failed_reads
            ),
            "native_kill_elapsed_seconds": (
                0.0 if kill_result is None else kill_result.elapsed_seconds
            ),
            "native_kill_candidate_diagnostics": (
                []
                if kill_result is None
                else [
                    {
                        "base": int(item.base_address),
                        "species": int(item.species_id),
                        "present_reads": int(item.present_reads),
                        "absent_reads": int(item.absent_reads),
                        "maximum_consecutive_absence": int(
                            item.maximum_consecutive_absence
                        ),
                        "minimum_seen_hp": item.minimum_seen_hp,
                        "last_seen_hp": item.last_seen_hp,
                        "initial_hp": item.initial_hp,
                        "zero_hp_reads": int(item.zero_hp_reads),
                        "maximum_consecutive_zero_hp": int(
                            item.maximum_consecutive_zero_hp
                        ),
                        "hp_decreased": bool(item.hp_decreased),
                        "confirmed": bool(item.confirmed),
                    }
                    for item in kill_result.diagnostics
                ]
            ),
            "eva_available": eva_available,
            "executed_action": int(
                action if executed_action is None else executed_action
            ),
            "executed_action_name": (
                action.name if executed_action is None else executed_action.name
            ),
            "jump_requested": bool(jump_requested),
            "jump_available": bool(jump_available),
            "jump_reward_available": bool(jump_reward_available),
            "jump_rewarded": bool(jump_requested and jump_reward_available),
            "jump_performed": bool(jump_performed),
            "jump_cooldown_fraction": float(
                self._jump_cooldown_fraction(self._clock())
            ),
            "player_displacement_cells": float(displacement_cells),
            "contact": bool(contact),
            "teleport_suspected": bool(
                False if teleport is None else teleport.suspected
            ),
            "teleport_confirmed": bool(
                False if teleport is None else teleport.confirmed
            ),
            "teleport_source_in_mapped_area": bool(
                False
                if teleport_response is None
                else teleport_response.source_in_mapped_teleport_area
            ),
            "unexpected_teleport": bool(
                teleport is not None
                and teleport.confirmed
                and teleport_response is not None
                and not teleport_response.source_in_mapped_teleport_area
            ),
            "teleport_recovery_pulse_attempted": bool(
                False
                if teleport_response is None
                else teleport_response.pulse_attempted
            ),
            "teleport_recovery_pulse_completed": bool(
                False
                if teleport_response is None
                else teleport_response.pulse_completed
            ),
            "teleport_recovery_pulse_seconds": float(
                0.0
                if teleport_response is None
                else teleport_response.pulse_seconds
            ),
            "teleport_recovery_pulse_error": (
                None
                if teleport_response is None
                else teleport_response.pulse_error
            ),
            "teleport_reason": (
                None if teleport is None else teleport.reason
            ),
            "teleport_effective_threshold_cells": (
                None if teleport is None else teleport.effective_threshold_cells
            ),
            "teleport_elapsed_seconds": (
                None if teleport is None else teleport.elapsed_seconds
            ),
            "teleport_thresholds_cells": (
                [] if teleport is None else list(teleport.thresholds_cells)
            ),
            "teleport_displacements_cells": (
                [] if teleport is None else list(teleport.displacements_cells)
            ),
            "teleport_destination_spread_cells": (
                None if teleport is None else teleport.destination_spread_cells
            ),
            "teleport_player_identity_stable": (
                None if teleport is None else teleport.player_identity_stable
            ),
            "teleport_player_identity_changed": (
                None if teleport is None else teleport.player_identity_changed
            ),
            "teleport_usable_sample_found": (
                None if teleport is None else teleport.usable_sample_found
            ),
            "teleport_selected_sample_index": (
                None if teleport is None else teleport.selected_sample_index
            ),
            "teleport_selected_map_cell": (
                None
                if teleport is None or not teleport.native_positions
                else self.map_context.native_to_layout_cell(
                    teleport.native_positions[teleport.selected_sample_index][0],
                    teleport.native_positions[teleport.selected_sample_index][2],
                )
            ),
            "teleport_before_player_base": (
                None if teleport is None else teleport.before_player_base
            ),
            "teleport_before_pointer_slot": (
                None if teleport is None else teleport.before_pointer_slot
            ),
            "teleport_before_native_position": (
                None
                if teleport is None
                else list(teleport.before_native_position)
            ),
            "teleport_before_pose_timestamp": (
                None if teleport is None else teleport.before_pose_timestamp
            ),
            "teleport_player_bases": (
                [] if teleport is None else list(teleport.player_bases)
            ),
            "teleport_pointer_slots": (
                [] if teleport is None else list(teleport.pointer_slots)
            ),
            "teleport_native_positions": (
                []
                if teleport is None
                else [list(position) for position in teleport.native_positions]
            ),
            "teleport_pose_timestamps": (
                [] if teleport is None else list(teleport.pose_timestamps)
            ),
            "player_native_position": (
                float(features.world.player_pose.x),
                float(features.world.player_pose.y),
                float(features.world.player_pose.z),
            ),
            "player_heading_degrees": (
                None
                if features.world.player_pose.heading_degrees is None
                else float(features.world.player_pose.heading_degrees)
            ),
            "pointer_mode": str(features.world.pointer_snapshot.mode),
            "pointer_generation": int(features.world.pointer_snapshot.generation),
            "ocr_outcome": None if ocr is None else ocr.outcome.value,
            "ocr_value": None if ocr is None else ocr.value,
            "ocr_previous": None if ocr is None else ocr.previous,
            "ocr_delta": None if ocr is None else ocr.delta,
            "visible_actors": features.built.visible_actor_count,
            "native_cached_actor_slots": int(
                getattr(actor_diagnostics, "discovered_slots", 0)
            ),
            "native_runtime_promoted_slots": int(
                getattr(actor_diagnostics, "runtime_promoted_slots", 0)
            ),
            "native_pending_actor_slot_probes": int(
                getattr(actor_diagnostics, "pending_actor_slot_probes", 0)
            ),
            "native_actor_source": str(
                getattr(actor_diagnostics, "actor_source", "unknown")
            ),
            "native_authoritative_relation_offset": getattr(
                actor_diagnostics, "authoritative_relation_offset", None
            ),
            "native_authoritative_relation_value": getattr(
                actor_diagnostics, "authoritative_relation_value", None
            ),
            "native_authoritative_relation_validated": bool(
                getattr(
                    actor_diagnostics,
                    "authoritative_relation_validated",
                    False,
                )
            ),
            "native_authoritative_species_counts": [
                [int(species), int(count)]
                for species, count in tuple(
                    getattr(
                        actor_diagnostics,
                        "authoritative_species_counts",
                        (),
                    )
                )
            ],
            "native_authoritative_refreshes": int(
                getattr(actor_diagnostics, "authoritative_refreshes", 0)
            ),
            "native_authoritative_refresh_failures": int(
                getattr(
                    actor_diagnostics,
                    "authoritative_refresh_failures",
                    0,
                )
            ),
            "native_authoritative_last_error": getattr(
                actor_diagnostics, "authoritative_last_error", None
            ),
            "native_active_species_offset": getattr(
                actor_diagnostics, "active_species_offset", None
            ),
            "native_active_species_validated": bool(
                getattr(actor_diagnostics, "active_species_validated", False)
            ),
            "native_active_species_candidates": [
                {
                    "offset": int(item[0]),
                    "living_matches": int(item[1]),
                    "living_samples": int(item[2]),
                    "zero_hp_matches": int(item[3]),
                    "zero_hp_samples": int(item[4]),
                    "validated": bool(item[5]),
                }
                for item in tuple(
                    getattr(actor_diagnostics, "active_species_candidates", ())
                )
            ],
            "eva_actors": features.built.eva_actor_count,
            "direct_clear_fraction": features.built.direct_clear_fraction,
            "held_movement": (
                None
                if self.control.held_movement is None
                else self.control.held_movement.name
            ),
            "map_name": self.map_context.map_name,
            "map_hash": self.map_context.content_hash,
            "map_cell": features.player_cell,
            "teleport_distance_cells": features.forbidden_distance_cells,
            "reward_components": (
                {} if reward is None else reward.components.as_dict()
            ),
            "session_ended": outcome.should_stop_session,
            "session_classification": outcome.classification.value,
            "session_end_reason": outcome.reason.value,
            "session_end_policy_caused": outcome.policy_caused,
            "session_detail": detail or outcome.detail,
        }

    def _seal(self, observation: FloatArray | None = None) -> None:
        try:
            self.control.release()
        finally:
            if observation is not None:
                terminal = np.asarray(observation, dtype=np.float32).copy()
                terminal.setflags(write=False)
                self._terminal_observation = terminal
            if self._state is not FarmingEnvironmentState.CLOSED:
                self._state = FarmingEnvironmentState.SEALED

    def close(self) -> None:
        if self._state is FarmingEnvironmentState.CLOSED:
            return
        try:
            self.control.close()
        finally:
            self._state = FarmingEnvironmentState.CLOSED

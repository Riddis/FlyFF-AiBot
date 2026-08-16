from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from farming.actions import (
    ACTION_NAMES,
    POLICY_ACTION_NVECS,
    FarmingAction,
    FarmingEvent,
    SteeringAction,
)
from farming.map_features import DirectPathState
from farming.observation import (
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_ID,
    ActorObservation,
    ObservationBuilder,
    ObservationFrame,
    ObservationScales,
    PlayerObservation,
)

from .map_model import MapModel
from .schema import (
    RecordedFrame,
    RecordingArchive,
    direct_movement_provenance_source,
    recording_sha256,
    unique_recording_paths,
    validate_recording_contract,
)
from .world_model import movement_action_from_mask, steering_action_from_mask

_DIRECT_PATH_LIMIT = 96


def _frame_observation(
    frame: RecordedFrame,
    previous: RecordedFrame | None,
    *,
    map_model: MapModel,
    builder: ObservationBuilder,
    last_eva_ms: int,
    last_jump_ms: int,
    previous_action: FarmingAction,
    held_movement: FarmingAction | None,
) -> tuple[np.ndarray, float]:
    player_cell = map_model.native_to_layout_cell(frame.player_x, frame.player_z)
    player_layout = map_model.native_to_layout_cells(frame.player_x, frame.player_z)
    candidates: list[tuple[float, object, float, float]] = []
    for actor in frame.actors:
        if not actor.living:
            continue
        dx = (actor.x - frame.player_x) / map_model.native_units_per_cell
        dz = (actor.z - frame.player_z) / map_model.native_units_per_cell
        distance = math.hypot(dx, dz)
        if distance <= builder.scales.vision_radius_cells:
            candidates.append((distance, actor, dx, dz))
    candidates.sort(key=lambda item: (item[0], item[1].base))
    direct_ids = {item[1].base for item in candidates[:_DIRECT_PATH_LIMIT]}
    geodesic_field = (
        map_model.features.bounded_geodesic_field(
            player_cell,
            maximum_distance_cells=builder.scales.vision_radius_cells * 1.5,
        )
        if player_cell is not None
        else {}
    )

    actors: list[ActorObservation] = []
    for _distance, actor, dx, dz in candidates:
        layout = map_model.native_to_layout_cells(actor.x, actor.z)
        actor_cell = map_model.native_to_layout_cell(actor.x, actor.z)
        geodesic = math.inf if actor_cell is None else geodesic_field.get(actor_cell, math.inf)
        direct_path = DirectPathState.UNKNOWN
        if actor.base in direct_ids:
            direct_path = map_model.features.direct_path_state(player_cell, actor_cell)
        actors.append(
            ActorObservation(
                actor_id=actor.base,
                legacy_dx_cells=layout[0] - player_layout[0],
                legacy_dy_cells=layout[1] - player_layout[1],
                direct_dx_cells=dx,
                direct_dz_cells=dz,
                geodesic_cells=geodesic,
                direct_path=direct_path,
            )
        )
    displacement = 0.0
    if previous is not None:
        displacement = math.hypot(
            frame.player_x - previous.player_x,
            frame.player_z - previous.player_z,
        ) / map_model.native_units_per_cell
    normalized_x, normalized_z = map_model.features.normalized_position(player_cell)
    eva_fraction = float(np.clip((frame.elapsed_ms - last_eva_ms) / 2000.0, 0.0, 1.0))
    jump_fraction = float(np.clip((frame.elapsed_ms - last_jump_ms) / 2000.0, 0.0, 1.0))
    vector = builder.build(
        ObservationFrame(
            player=PlayerObservation(
                normalized_x=normalized_x,
                normalized_z=normalized_z,
                heading_radians=frame.heading_radians,
                eva_cooldown_fraction=eva_fraction,
                displacement_cells=displacement,
                contact=False,
                held_movement=held_movement,
                last_policy_action=previous_action,
                jump_cooldown_fraction=jump_fraction,
                map_available=True,
            ),
            actors=tuple(actors),
            local_map=map_model.features.local_crop(player_cell, side=11),
            context_map=map_model.features.context_crop(player_cell),
        )
    ).vector
    # Raw (unclipped, unencoded) displacement -- kept separate from the
    # observation vector's own bipolar-encoded, maximum_displacement_cells
    # -clipped copy of the same quantity, so callers building the
    # navigation-sidecar history (NavigationStepEvidence.displacement_cells)
    # get an exact value rather than round-tripping a lossy encoding.
    # `contact` above is always False: real recordings carry no collision
    # ground truth (see export_demonstrations' docstring) -- never invert
    # this encoded field to approximate contact either.
    return vector, displacement


def export_demonstrations(
    recording_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    map_model: MapModel | None = None,
    maximum_samples: int | None = None,
    eva_only_recording_paths: Iterable[str | Path] = (),
    allow_legacy_direct_provenance: bool = False,
) -> Path:
    """Build a 923-value-observation BC dataset from recording archives.

    Saved arrays include `session_index`/`elapsed_ms` (already in per-session
    sequential order) and `displacement_cells` (exact, unencoded per-tick
    real displacement, derived from consecutive frame positions) -- together
    sufficient to reconstruct NavigationHistoryWrapper-style
    NavigationStepEvidence sequences for the recent_progress sidecar
    feature. There is no equivalent for `contact`: real recordings carry no
    collision ground truth (no "was this movement blocked" signal exists in
    the recording format), so every sample's `contact` is fixed False here,
    and recent_contact cannot be faithfully reconstructed from this dataset
    -- feed a documented-neutral placeholder for it during human-recording
    bootstrap, do not infer it from displacement shortfall (that would be
    invented data, not a measured one).
    """
    map_data = map_model or MapModel.load()
    builder = ObservationBuilder(ObservationScales())
    observations: list[np.ndarray] = []
    actions: list[tuple[int, int]] = []
    legacy_actions: list[int] = []
    steering_label_valid: list[bool] = []
    event_label_valid: list[bool] = []
    sessions: list[int] = []
    elapsed_ms: list[int] = []
    displacement_cells: list[float] = []
    direct_paths = unique_recording_paths(recording_paths)
    eva_only_paths = unique_recording_paths(eva_only_recording_paths)
    source_paths = unique_recording_paths((*direct_paths, *eva_only_paths))
    eva_only = {path.resolve() for path in eva_only_paths}
    source_hashes: list[str] = []
    source_roles: list[str] = []
    source_provenance: list[str] = []
    contract_warnings: list[str] = []
    for session_index, path in enumerate(source_paths):
        archive = RecordingArchive(path)
        source_hash = recording_sha256(path)
        source_hashes.append(source_hash)
        role = "eva_only" if path.resolve() in eva_only else "direct_keyboard"
        source_roles.append(role)
        provenance_source = direct_movement_provenance_source(
            archive.manifest, recording_hash=source_hash
        )
        source_provenance.append(provenance_source or "none_required_for_eva_only")
        if (
            role == "direct_keyboard"
            and provenance_source is None
            and not allow_legacy_direct_provenance
        ):
            raise ValueError(
                f"{path}: direct movement export requires recorder provenance "
                "recording_role='direct_keyboard_demonstration' and "
                "movement_control_scheme='keyboard_wasd'; use the legacy override "
                "only after manually verifying an older archive"
            )
        if role == "direct_keyboard" and provenance_source is None:
            contract_warnings.append(
                f"{path.name}: direct movement labels accepted via explicit legacy provenance override"
            )
        elif role == "direct_keyboard" and provenance_source == "sha256_attestation_registry":
            contract_warnings.append(
                f"{path.name}: direct WASD provenance supplied by SHA-256 attestation registry"
            )
        contract_warnings.extend(
            validate_recording_contract(
                archive,
                action_names=ACTION_NAMES,
                origin_native_x=map_data.origin_native_x,
                origin_native_z=map_data.origin_native_z,
                native_units_per_cell=map_data.native_units_per_cell,
                observation_schema_id=OBSERVATION_SCHEMA_ID,
                observation_schema_hash=OBSERVATION_SCHEMA_HASH,
            )
        )
        previous: RecordedFrame | None = None
        last_eva = -2_000_000_000
        last_jump = -2_000_000_000
        previous_action = FarmingAction.RUN_FORWARD
        held_movement: FarmingAction | None = FarmingAction.RUN_FORWARD
        for frame in archive.frames():
            mask_action = movement_action_from_mask(frame.key_mask)
            # movement_action_from_mask collapses any forward+jump combination
            # to legacy action 4 regardless of concurrent left/right bits --
            # correct for that legacy scalar, but it must never be used to
            # derive held/steering state here, or a jump tap would silently
            # erase the concurrently-held steering key. steering_action_from_mask
            # reads only the forward/left/right bits, independent of jump.
            mask_steering = steering_action_from_mask(frame.key_mask)
            action_is_eva = frame.action == int(FarmingAction.CAST_EVA)
            action_is_verified_movement = bool(
                frame.action in (0, 1, 2, 4) and mask_action == frame.action
            )
            # A movement label is valid only when the keyboard mask proves the
            # exact low-level action. Coordinate displacement is never used to
            # infer WASD from click-to-move. World recordings can be supplied
            # explicitly as EVA-only sources.
            recognized = action_is_eva or (
                role == "direct_keyboard" and action_is_verified_movement
            )
            frame_held = held_movement
            if mask_steering is not None:
                frame_held = (
                    FarmingAction.RUN_FORWARD,
                    FarmingAction.RUN_FORWARD_LEFT,
                    FarmingAction.RUN_FORWARD_RIGHT,
                )[mask_steering]
            if frame.phase == 1 and frame.focused and recognized:
                observation_vector, raw_displacement = _frame_observation(
                    frame,
                    previous,
                    map_model=map_data,
                    builder=builder,
                    last_eva_ms=last_eva,
                    last_jump_ms=last_jump,
                    previous_action=previous_action,
                    held_movement=frame_held,
                )
                observations.append(observation_vector)
                displacement_cells.append(raw_displacement)
                steering = {
                    FarmingAction.RUN_FORWARD: SteeringAction.STRAIGHT,
                    FarmingAction.RUN_FORWARD_LEFT: SteeringAction.LEFT,
                    FarmingAction.RUN_FORWARD_RIGHT: SteeringAction.RIGHT,
                }.get(frame_held or FarmingAction.RUN_FORWARD, SteeringAction.STRAIGHT)
                event = (
                    FarmingEvent.CAST_EVA
                    if frame.action == int(FarmingAction.CAST_EVA)
                    else (
                        FarmingEvent.JUMP
                        if frame.action == int(FarmingAction.RUN_FORWARD_JUMP)
                        else FarmingEvent.NONE
                    )
                )
                actions.append((int(steering), int(event)))
                legacy_actions.append(int(frame.action))
                # Click-to-move/EVA-only archives contribute only the observed
                # event label. Their steering cannot be inferred from coordinate
                # deltas and must never supervise the steering head.
                steering_label_valid.append(role == "direct_keyboard")
                event_label_valid.append(True)
                sessions.append(session_index)
                elapsed_ms.append(frame.elapsed_ms)
            if recognized:
                current_action = FarmingAction(frame.action)
                if current_action is FarmingAction.CAST_EVA:
                    last_eva = frame.elapsed_ms
                if current_action is FarmingAction.RUN_FORWARD_JUMP:
                    last_jump = frame.elapsed_ms
                if current_action.is_movement and mask_steering is not None:
                    held_movement = (
                        FarmingAction.RUN_FORWARD,
                        FarmingAction.RUN_FORWARD_LEFT,
                        FarmingAction.RUN_FORWARD_RIGHT,
                    )[mask_steering]
                previous_action = current_action
            elif mask_steering is not None:
                held_movement = (
                    FarmingAction.RUN_FORWARD,
                    FarmingAction.RUN_FORWARD_LEFT,
                    FarmingAction.RUN_FORWARD_RIGHT,
                )[mask_steering]
            previous = frame
            if maximum_samples is not None and len(observations) >= maximum_samples:
                break
        if maximum_samples is not None and len(observations) >= maximum_samples:
            break
    if not observations:
        raise ValueError("No focused farming frames with recognized actions were found")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        legacy_actions=np.asarray(legacy_actions, dtype=np.int64),
        steering_label_valid=np.asarray(steering_label_valid, dtype=np.bool_),
        event_label_valid=np.asarray(event_label_valid, dtype=np.bool_),
        action_contract_id=np.asarray(["latched-forward-factorized-steering-event-v1"]),
        action_nvec=np.asarray(POLICY_ACTION_NVECS, dtype=np.int64),
        session_index=np.asarray(sessions, dtype=np.int32),
        elapsed_ms=np.asarray(elapsed_ms, dtype=np.int64),
        displacement_cells=np.asarray(displacement_cells, dtype=np.float32),
        observation_schema_id=np.asarray([builder.schema_id]),
        observation_schema_hash=np.asarray([builder.schema_hash]),
        source_recording_sha256=np.asarray(source_hashes),
        source_recording_role=np.asarray(source_roles),
        source_recording_provenance=np.asarray(source_provenance),
        contract_warnings=np.asarray(tuple(dict.fromkeys(contract_warnings))),
    )
    return output

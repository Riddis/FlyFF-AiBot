from __future__ import annotations

from pathlib import Path

from position.IndependentNativeReader import IndependentActorSlotRead
from recorder.config import RecorderConfig
from recorder.format import PackedStreamWriter, read_packed_stream
from recorder.keyboard import (
    EVA_BIT,
    FORWARD_BIT,
    SUPPORTED_EVA_HOTKEYS,
    virtual_key_for_hotkey,
)
from recorder.movement_classification import MovementControlClassifier
from recorder.lifecycle import LifecycleTracker


def actor(
    *,
    hp: int,
    state: str,
    x: float = 10.0,
    z: float = 20.0,
    species: int = 944,
    target_species: bool = True,
) -> IndependentActorSlotRead:
    return IndependentActorSlotRead(
        base=0x123400,
        species=species,
        hp=hp,
        x=x,
        y=0.0,
        z=z,
        active_species=species,
        active_matches_species=True,
        target_species=target_species,
        state=state,
        distance_native=4.0,
    )


def test_config_defaults_are_valid() -> None:
    config = RecorderConfig()
    config.validate()
    assert config.monster_hp == {944: 400236}
    assert config.selected_species == {944, 948}
    assert config.frame_interval_seconds == 0.2
    assert config.presence_clear_confirmation_samples == 3
    assert config.presence_cold_poll_batch_size == 1024
    assert config.presence_cold_verification_batch_size == 256
    assert not hasattr(config, "recording_role")
    assert not hasattr(config, "movement_control_scheme")


def test_legacy_hidden_classification_keys_are_ignored(tmp_path: Path) -> None:
    config_path = tmp_path / "recorder_config.json"
    config_path.write_text(
        '{"recording_role":"world_model",'
        '"movement_control_scheme":"unknown",'
        '"selected_species_ids":[944,948]}',
        encoding="utf-8",
    )
    config = RecorderConfig.load(config_path)
    config.validate()
    assert not hasattr(config, "recording_role")
    assert not hasattr(config, "movement_control_scheme")


def _feed_movement(classifier: MovementControlClassifier, masks: list[int]) -> None:
    x = 0.0
    classifier.observe(x=x, z=0.0, focused=True, key_mask=masks[0])
    for mask in masks[1:]:
        x += 1.0
        classifier.observe(x=x, z=0.0, focused=True, key_mask=mask)


def test_movement_control_is_classified_automatically() -> None:
    keyboard = MovementControlClassifier(minimum_distance_native=5.0)
    _feed_movement(keyboard, [FORWARD_BIT] * 25)
    keyboard_report = keyboard.report()
    assert keyboard_report.scheme == "keyboard_wasd"
    assert keyboard_report.direct_movement_labels_allowed is True

    click = MovementControlClassifier(minimum_distance_native=5.0)
    _feed_movement(click, [0] * 25)
    click_report = click.report()
    assert click_report.scheme == "click_to_move"
    assert click_report.direct_movement_labels_allowed is False


def test_forward_held_through_eva_still_explains_movement() -> None:
    classifier = MovementControlClassifier(minimum_distance_native=5.0)
    masks = [FORWARD_BIT | (EVA_BIT if index % 5 == 0 else 0) for index in range(25)]
    _feed_movement(classifier, masks)
    report = classifier.report()
    assert report.scheme == "keyboard_wasd"
    assert report.keyboard_explained_distance_ratio == 1.0


def test_packed_stream_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "stream.msgpack.gz"
    with PackedStreamWriter(path, header={"name": "test"}) as writer:
        writer.write(["frame", 1, 2, 3])
        writer.write({"event": "done"})
    values = list(read_packed_stream(path))
    assert values[0]["type"] == "header"
    assert values[1] == ["frame", 1, 2, 3]
    assert values[2] == {"event": "done"}


def test_hotkey_choices_include_function_and_number_row_keys() -> None:
    assert "F1" in SUPPORTED_EVA_HOTKEYS
    assert "F12" in SUPPORTED_EVA_HOTKEYS
    assert tuple(str(i) for i in range(1, 10)) == SUPPORTED_EVA_HOTKEYS[-9:]
    assert virtual_key_for_hotkey("F1") == 0x70
    assert virtual_key_for_hotkey("F12") == 0x7B
    assert virtual_key_for_hotkey("2") == ord("2")


def test_lifecycle_tracks_same_slot_death_as_respawn_candidate() -> None:
    tracker = LifecycleTracker(kill_radius=80.0)
    quantize = lambda value: int(round(value * 20.0))
    assert tracker.update(
        [actor(hp=400236, state="living")],
        elapsed_ms=0,
        player_x_q=0,
        player_z_q=0,
        phase=0,
        quantize=quantize,
    ) == []
    deaths = tracker.update(
        [actor(hp=0, state="dead")],
        elapsed_ms=25,
        player_x_q=0,
        player_z_q=0,
        phase=1,
        quantize=quantize,
    )
    assert deaths[0][0] == "death"
    assert deaths[0][10] is True
    candidates = tracker.update(
        [actor(hp=400236, state="living", x=50.0, z=60.0)],
        elapsed_ms=525,
        player_x_q=0,
        player_z_q=0,
        phase=1,
        quantize=quantize,
    )
    assert candidates[0][0] == "respawn_candidate"
    assert candidates[0][12] == 500
    assert tracker.probable_kills == 1
    assert tracker.respawn_candidates == 1


def test_streamed_target_slot_is_an_appearance_not_a_respawn() -> None:
    tracker = LifecycleTracker(kill_radius=80.0)
    quantize = lambda value: int(round(value * 20.0))
    tracker.update(
        [actor(hp=400236, state="other_species", species=948, target_species=False)],
        elapsed_ms=0,
        player_x_q=0,
        player_z_q=0,
        phase=0,
        quantize=quantize,
    )
    events = tracker.update(
        [actor(hp=400236, state="living", species=944, target_species=True)],
        elapsed_ms=25,
        player_x_q=0,
        player_z_q=0,
        phase=0,
        quantize=quantize,
    )
    assert [event[0] for event in events] == ["reuse", "target_appearance"]
    assert tracker.target_appearances == 1
    assert tracker.respawn_candidates == 0


def test_lifecycle_tracks_both_selected_tower_species() -> None:
    tracker = LifecycleTracker(kill_radius=80.0)
    quantize = lambda value: int(round(value * 20.0))
    dantalian = actor(
        hp=400236,
        state="living",
        species=948,
        target_species=True,
    )
    tracker.update(
        [dantalian],
        elapsed_ms=0,
        player_x_q=0,
        player_z_q=0,
        phase=1,
        quantize=quantize,
    )
    events = tracker.update(
        [actor(hp=0, state="dead", species=948, target_species=True)],
        elapsed_ms=25,
        player_x_q=0,
        player_z_q=0,
        phase=1,
        quantize=quantize,
    )
    assert events[0][0] == "death"
    assert events[0][3] == 948
    assert tracker.deaths_by_species == {948: 1}
    assert tracker.probable_kills_by_species == {948: 1}


def test_selected_species_are_separate_from_full_hp_anchors(tmp_path: Path) -> None:
    config_path = tmp_path / "recorder_config.json"
    config_path.write_text(
        '{"monster_hp_by_species":{"944":400236},"selected_species_ids":[944,948]}',
        encoding="utf-8",
    )
    config = RecorderConfig.load(config_path)
    config.validate()
    assert config.monster_hp == {944: 400236}
    assert config.selected_species == {944, 948}


def test_installer_preserves_recordings() -> None:
    installer = (
        Path(__file__).resolve().parents[1] / "FlyffFarmingRecorderInstaller.iss"
    ).read_text(encoding="utf-8")
    assert "[UninstallDelete]" not in installer
    assert "{userdocs}\\FlyffFarmingRecorder" not in installer


def test_gui_starts_directly_in_farming_without_exploration_controls() -> None:
    from recorder.gui import RecorderGui

    source = (Path(__file__).resolve().parents[1] / "recorder" / "gui.py").read_text(
        encoding="utf-8"
    )
    assert "Test Hotkey" not in source
    assert "hotkey_tested" not in source
    assert "farming_button" not in source
    assert "discovery_progress" not in source
    assert "map-coverage" not in source
    assert "exploration" not in source.casefold()
    assert "Farm normally" in source
    assert RecorderGui is not None


def test_background_rediscovery_is_adaptive_and_movement_triggered() -> None:
    config = RecorderConfig()
    assert config.rediscovery_stable_scan_interval_seconds == 2.0
    assert config.rediscovery_stable_scan_count == 3
    assert config.rediscovery_movement_trigger_native == 12.0
    source = (Path(__file__).resolve().parents[1] / "recorder" / "session.py").read_text(
        encoding="utf-8"
    )
    assert "if rediscovery_thread is None and monotonic() >= next_rediscovery_at:" in source
    assert "consecutive_stable_scans" in source
    assert "rediscovery_stable_scan_interval_seconds" in source
    assert "rediscovery_movement_trigger_native" in source
    assert "Player entered a new area; actor rediscovery started immediately." in source
    assert "rediscover_selected_layout_monsters" in source
    assert "species_ids=self.config.selected_species" in source
    assert "allowed_species=self.config.selected_species" in source


def test_active_field_profiler_recovers_instantiated_duplicate_without_using_zero_hp_as_negative() -> None:
    import struct

    from recorder.active_field_profiler import ActiveFieldProfiler

    class Memory:
        def __init__(self) -> None:
            self.blocks: dict[int, bytearray] = {}

        def read(self, address: int, size: int) -> bytes:
            for base, block in self.blocks.items():
                if base <= address and address + size <= base + len(block):
                    start = address - base
                    return bytes(block[start : start + size])
            raise OSError("unmapped")

    memory = Memory()
    stride = 0x2008
    active_offset = 0x1DBC
    always_duplicate = 0x500
    bases = [0x200000 + index * 0x4000 for index in range(20)]
    states = []
    for index, base in enumerate(bases):
        species = 944 if index < 10 else 948
        block = bytearray(stride)
        struct.pack_into("<i", block, 0x174, species)
        struct.pack_into("<i", block, active_offset, species)
        struct.pack_into("<i", block, always_duplicate, species)
        memory.blocks[base] = block
        states.append(
            IndependentActorSlotRead(
                base=base,
                species=species,
                hp=400236,
                x=10.0,
                y=0.0,
                z=20.0,
                active_species=species,
                active_matches_species=True,
                target_species=species == 944,
                state="living",
                distance_native=4.0,
            )
        )

    profiler = ActiveFieldProfiler(
        memory,
        actor_stride=stride,
        object_span=0x4000,
        excluded_offsets={0x174, 0x81C, 0x160, 0x164, 0x168},
    )
    for sample in range(2):
        assert profiler.sample_live_states(
            states,
            elapsed_ms=sample * 500,
            maximum_samples=20,
            maximum_distance_native=80.0,
        ) == 20

    # A zero-HP corpse can remain instantiated. The candidate is allowed to keep
    # matching here and is not rejected for doing so.
    profiler.observe_event(
        ["death", 1600, bases[0], 944],
        elapsed_ms=1600,
    )

    # Move away while the ordinary species/HP/position remain stale. The true
    # instantiated field clears, while a generic species duplicate remains and
    # must not validate.
    far_states = []
    for index, base in enumerate(bases):
        species = 944 if index < 10 else 948
        struct.pack_into("<i", memory.blocks[base], active_offset, 0)
        far_states.append(
            IndependentActorSlotRead(
                base=base,
                species=species,
                hp=400236,
                x=10.0,
                y=0.0,
                z=20.0,
                active_species=0,
                active_matches_species=False,
                target_species=species == 944,
                state="living",
                distance_native=200.0,
            )
        )
    profiler.sample_live_states(
        far_states,
        elapsed_ms=5000,
        maximum_samples=20,
        maximum_distance_native=80.0,
    )
    assert profiler.sample_dormant_states(
        far_states,
        elapsed_ms=5000,
        minimum_distance_native=160.0,
        stable_milliseconds=3000,
        after_near_milliseconds=2000,
        maximum_samples=20,
    ) == 20

    # Same-slot reappearance is intentionally absent. Actor addresses are reusable
    # pool slots, so strong live and dormant-transition evidence must be enough.
    report = profiler.report()
    assert report["recommended_offset"] == "0x1DBC"
    candidate = next(
        item for item in report["candidates"] if item["offset_hex"] == "0x1DBC"
    )
    assert candidate["validated"] is True
    assert candidate["zero_hp_match_ratio"] == 1.0
    stale = next(
        item for item in report["candidates"] if item["offset_hex"] == "0x500"
    )
    assert stale["validated"] is False
    assert report["historical_offsets"]["0x217C"] is None


def test_active_field_profiler_report_keeps_promoted_offset_after_evidence_drifts() -> None:
    """A field promoted mid-session must stay reported as validated even if
    later cumulative diagnostic evidence alone would no longer pass the
    strict gate. The native reader used the promoted offset for the rest of
    the session regardless of how the profiler's own running ratios moved
    afterward, so the exported report, and the final summary log built from
    it, must not contradict that by claiming nothing was ever proven."""

    import struct

    from recorder.active_field_profiler import ActiveFieldProfiler

    class Memory:
        def __init__(self) -> None:
            self.blocks: dict[int, bytearray] = {}

        def read(self, address: int, size: int) -> bytes:
            for base, block in self.blocks.items():
                if base <= address and address + size <= base + len(block):
                    start = address - base
                    return bytes(block[start : start + size])
            raise OSError("unmapped")

    memory = Memory()
    stride = 0x2008
    active_offset = 0x1DBC
    bases = [0x300000 + index * 0x4000 for index in range(20)]
    states = []
    for index, base in enumerate(bases):
        species = 944 if index < 10 else 948
        block = bytearray(stride)
        struct.pack_into("<i", block, 0x174, species)
        struct.pack_into("<i", block, active_offset, species)
        memory.blocks[base] = block
        states.append(
            IndependentActorSlotRead(
                base=base,
                species=species,
                hp=400236,
                x=10.0,
                y=0.0,
                z=20.0,
                active_species=species,
                active_matches_species=True,
                target_species=species == 944,
                state="living",
                distance_native=4.0,
            )
        )

    profiler = ActiveFieldProfiler(
        memory,
        actor_stride=stride,
        object_span=0x4000,
        excluded_offsets={0x174, 0x81C, 0x160, 0x164, 0x168},
    )
    for sample in range(2):
        profiler.sample_live_states(
            states, elapsed_ms=sample * 500, maximum_samples=20, maximum_distance_native=80.0
        )

    far_states = []
    for index, base in enumerate(bases):
        species = 944 if index < 10 else 948
        struct.pack_into("<i", memory.blocks[base], active_offset, 0)
        far_states.append(
            IndependentActorSlotRead(
                base=base,
                species=species,
                hp=400236,
                x=10.0,
                y=0.0,
                z=20.0,
                active_species=0,
                active_matches_species=False,
                target_species=species == 944,
                state="living",
                distance_native=200.0,
            )
        )
    profiler.sample_live_states(
        far_states, elapsed_ms=5000, maximum_samples=20, maximum_distance_native=80.0
    )
    profiler.sample_dormant_states(
        far_states,
        elapsed_ms=5000,
        minimum_distance_native=160.0,
        stable_milliseconds=3000,
        after_near_milliseconds=2000,
        maximum_samples=20,
    )

    before = profiler.report()
    assert before["recommended_offset"] == "0x1DBC"

    # Promotion happens here, mid-session, exactly as session.py does it.
    profiler.mark_promoted(0x1DBC)

    # More of the session elapses: the field stops clearing on later dormant
    # reads (e.g. noisier evidence far into a long recording), which alone
    # would flip `_OffsetEvidence.validated` back to False.
    for index, base in enumerate(bases):
        struct.pack_into("<i", memory.blocks[base], active_offset, far_states[index].species)
    for offset in range(0, 40):
        elapsed = 20_000 + offset * 3000
        for index, base in enumerate(bases):
            history = profiler._state_history[base]
            history.last_changed_ms = elapsed - 5000
            history.last_near_ms = elapsed - 4000
            history.last_dormant_sample_ms = -10_000_000
        profiler.sample_dormant_states(
            far_states,
            elapsed_ms=elapsed,
            minimum_distance_native=160.0,
            stable_milliseconds=3000,
            after_near_milliseconds=2000,
            maximum_samples=20,
        )

    live_evidence = profiler._evidence[0x1DBC]
    assert live_evidence.dormant_ratio > 0.10  # drifted past the strict gate
    assert live_evidence.validated is False  # confirms the drift really happened

    after = profiler.report()
    assert after["recommended_offset"] == "0x1DBC"
    assert "0x1DBC" in after["validated_offsets"]
    assert "0x1DBC" in after["promoted_offsets"]
    candidate = next(item for item in after["candidates"] if item["offset_hex"] == "0x1DBC")
    assert candidate["promoted_this_session"] is True


def test_recorder_profiles_and_uses_instantiated_field_as_verified_hint() -> None:
    source = (Path(__file__).resolve().parents[1] / "recorder" / "session.py").read_text(
        encoding="utf-8"
    )
    capture_source = (
        Path(__file__).resolve().parents[1] / "recorder" / "native_capture.py"
    ).read_text(encoding="utf-8")
    reader_source = (
        Path(__file__).resolve().parents[1] / "position" / "IndependentNativeReader.py"
    ).read_text(encoding="utf-8")
    assert 'RECORDER_VERSION = "1.11.0"' in source
    assert '"recording_provenance": recording_provenance' in source
    assert '"data_quality": {' in source
    assert '"Data-quality checkpoint: "' in source
    assert '"Final data classification: "' in source
    assert 'session_dir / "active_field_profile.json"' in source
    assert '"active_field_profile": "active_field_profile.json"' in source
    assert "enable_presence_optimized_sampling" in capture_source
    assert "restore_profile" in capture_source
    assert "persist_attached_profile" in capture_source
    assert "promote_validated_presence_offset" in source
    assert "MovementControlClassifier" in source
    assert "presence_cold_verification_batch_size" in capture_source
    assert "_scan_monsters_presence_optimized" in reader_source
    assert "rotating presence and full-verification batches" in reader_source


def test_recording_provenance_emits_recording_role_the_simulator_gate_requires() -> None:
    """simulator.schema.allows_direct_movement_labels only trusts an embedded
    manifest when recording_provenance.recording_role equals
    "direct_keyboard_demonstration" -- the two projects ship independently so
    this string is duplicated, not shared. A prior version of session.py
    never wrote recording_role at all, so no recorder-emitted archive could
    ever be recognized as demonstration-ready through the embedded path,
    regardless of how confidently its movement was classified."""

    source = (Path(__file__).resolve().parents[1] / "recorder" / "session.py").read_text(
        encoding="utf-8"
    )
    assert '"recording_role"' in source
    assert '"direct_keyboard_demonstration"' in source
    assert "active_profiler.mark_promoted(" in source


def test_presence_sampler_keeps_death_animation_and_cools_after_confirmed_clear() -> None:
    import math
    import struct
    from threading import RLock
    from types import SimpleNamespace

    from position.IndependentNativeReader import IndependentNativeReader, IndependentPlayerRead

    class Memory:
        def __init__(self) -> None:
            self.blocks: dict[int, bytearray] = {}

        def read(self, address: int, size: int) -> bytes:
            for base, block in self.blocks.items():
                if base <= address and address + size <= base + len(block):
                    start = address - base
                    return bytes(block[start : start + size])
            raise OSError("unmapped")

    memory = Memory()
    bases = tuple(0x300000 + index * 0x4000 for index in range(10))
    for index, base in enumerate(bases):
        block = bytearray(0x2008)
        struct.pack_into("<I", block, 0x1EF0, base)
        if index == 0:
            struct.pack_into("<i", block, 0x174, 944)
            struct.pack_into("<i", block, 0x81C, 400236)
            struct.pack_into("<f", block, 0x160, 10.0)
            struct.pack_into("<f", block, 0x164, 0.0)
            struct.pack_into("<f", block, 0x168, 20.0)
            struct.pack_into("<i", block, 0x1DCC, 944)
        memory.blocks[base] = block

    reader = object.__new__(IndependentNativeReader)
    reader._memory = memory
    reader.monster_targets = (
        SimpleNamespace(
            species_offset=0x174,
            x_offset=0x160,
            y_offset=0x164,
            z_offset=0x168,
            self_pointer_offsets=(0x1EF0,),
        ),
    )
    reader.monster_hp_offset = 0x81C
    reader.monster_active_species_offset = 0x1DBC
    reader._active_species_offset = None
    reader._actor_self_offsets = (0x1EF0,)
    reader.expected_full_hp_by_species = {944: 400236}
    reader.actor_stride = 0x2008
    reader._cache_lock = RLock()
    reader._actor_slots = bases
    reader._presence_species_offset = None
    reader._recovered_presence_species_offset = 0x1DCC
    reader._presence_species_validated = True
    reader._presence_sampling_requested = False
    reader._presence_selected_species = set()
    reader._presence_clear_confirmation_samples = 2
    reader._presence_cold_poll_batch_size = 4
    reader._presence_cold_verification_batch_size = 1
    reader._presence_dead_read_grace_seconds = 2.0
    reader._presence_last_states = {}
    reader._presence_clear_counts = {}
    reader._presence_hot_until = {}
    reader._presence_cold_poll_cursor = 0
    reader._presence_cold_verify_cursor = 0
    reader._presence_snapshots = 0
    reader._presence_reads = 0
    reader._presence_full_actor_reads = 0
    reader._presence_cold_verification_reads = 0
    reader._presence_last_hot_slots = 0
    reader._presence_last_cold_slots = 0
    reader.enable_presence_optimized_sampling(
        selected_species_ids={944, 948},
        clear_confirmation_samples=2,
        cold_poll_batch_size=4,
        cold_verification_batch_size=1,
        dead_read_grace_seconds=2.0,
    )
    player = IndependentPlayerRead(0x100000, 0x200000, 1000, 0.0, 0.0, 0.0)

    first = reader._scan_monsters_presence_optimized(
        player, allowed_species={944, 948}, vision_radius_native=None
    )
    assert first.living == 1
    assert len(first.actor_states) == 10
    assert reader.presence_sampler_diagnostics().full_actor_reads < 10

    struct.pack_into("<i", memory.blocks[bases[0]], 0x81C, 0)
    corpse = reader._scan_monsters_presence_optimized(
        player, allowed_species={944, 948}, vision_radius_native=None
    )
    assert corpse.zero_hp == 1
    assert corpse.actor_states[0].state == "dead"

    struct.pack_into("<i", memory.blocks[bases[0]], 0x1DCC, 0)
    first_clear = reader._scan_monsters_presence_optimized(
        player, allowed_species={944, 948}, vision_radius_native=None
    )
    assert first_clear.actor_states[0].state == "dead"
    second_clear = reader._scan_monsters_presence_optimized(
        player, allowed_species={944, 948}, vision_radius_native=None
    )
    assert second_clear.actor_states[0].target_species is False
    assert second_clear.actor_states[0].state == "empty"
    assert math.isfinite(second_clear.actor_states[0].x)


def test_initial_authoritative_relation_does_not_require_unloaded_selected_species() -> None:
    from position.AuthoritativeActorDiscovery import (
        RelationScanEvidence,
        _strong_anchor_relation_proven,
    )

    evidence = RelationScanEvidence(
        offset=0x16C,
        value=0x12345678,
        references=6863,
        unique_candidate_bases=6863,
        valid_actor_bases=486,
        exact_anchor_coverage=73,
        exact_anchor_total=73,
        selected_species_counts=((944, 486),),
        self_rejections=0,
        relation_rejections=0,
        species_rejections=0,
        hp_rejections=0,
        coordinate_rejections=0,
        unreadable_rejections=0,
        search_bytes_read=945 << 20,
        search_regions_read=1662,
    )
    assert _strong_anchor_relation_proven(
        evidence, exact_total=73, anchor_species={944}
    )
    assert 948 not in dict(evidence.selected_species_counts)


def test_map_exit_guards_are_accuracy_first() -> None:
    config = RecorderConfig()
    assert config.map_exit_max_distance_from_spawn_native == 1500.0
    assert config.map_exit_jump_native == 250.0
    assert config.map_exit_confirmation_samples == 2
    source = (Path(__file__).resolve().parents[1] / "recorder" / "session.py").read_text(
        encoding="utf-8"
    )
    assert 'stop_reason = "map_exit_detected"' in source
    assert '"session_boundary"' in source
    assert "outside_tower_map_bounds" in source
    assert "teleport_sized_position_jump" in source
    assert "continue" in source
    assert "before out-of-map memory could enter the dataset" in source


def test_quantizer_cannot_overflow_msgpack_on_stale_pointer_values() -> None:
    source = (Path(__file__).resolve().parents[1] / "recorder" / "session.py").read_text(
        encoding="utf-8"
    )
    assert "9_000_000_000_000_000_000" in source
    assert "astronomically large float" in source


def test_player_discovery_does_not_gate_on_monster_instantiated_field() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "position" / "NativeTraceTargets.py"
    ).read_text(encoding="utf-8")
    assert "exact_monster_bases = {int(item.base) for item in anchors}" in source
    assert "if base in exact_monster_bases:" in source
    assert "if species > 0 and active == species:" not in source
    assert "never be used as a mandatory player discriminator" in source

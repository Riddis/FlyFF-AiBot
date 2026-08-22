from __future__ import annotations

from farming.actions import FarmingEvent, SteeringAction
from farming.map_features import MapCellRisk
from simulator.recovery_controller import RecoveryConfig, RecoveryController, RecoveryState


class _FakeMap:
    """Always reports SAFE everywhere, so direction choice falls back to the
    controller's deterministic alternation -- this test targets the state
    machine, not the clearance sampler (covered separately)."""

    native_units_per_cell = 10.0

    class features:
        @staticmethod
        def cell_risk(cell):
            return MapCellRisk.SAFE

    @staticmethod
    def native_to_layout_cell(x, z):
        return (0, 0)


def _config(**overrides):
    base = dict(
        history_window=10,
        no_progress_displacement_threshold=0.15,
        no_progress_ticks_required=8,
        min_contacts_in_window=2,
        debounce_ticks=5,
        escape_turn_ticks=3,
        escape_progress_check_ticks=2,
        escape_progress_threshold_cells=1.0,
        max_escape_attempts=2,
    )
    base.update(overrides)
    return RecoveryConfig(**base)


def test_normal_progress_never_triggers_recovery() -> None:
    controller = RecoveryController(_config())
    for tick in range(50):
        steering, event = controller.step(
            tick=tick, player_x=0.0, player_z=0.0, heading=0.0,
            displacement_this_tick=1.0, contact_this_tick=False,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.NONE),
        )
        assert controller.state == RecoveryState.NORMAL
        assert steering == int(SteeringAction.STRAIGHT)
    assert controller.interventions == []


def test_sustained_no_progress_with_contacts_triggers_recovery() -> None:
    controller = RecoveryController(_config())
    triggered = False
    for tick in range(30):
        steering, event = controller.step(
            tick=tick, player_x=0.0, player_z=0.0, heading=0.0,
            displacement_this_tick=0.0, contact_this_tick=True,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.NONE),
        )
        if controller.state == RecoveryState.RECOVERING:
            triggered = True
            break
    assert triggered


def test_eva_casting_does_not_falsely_trigger_recovery() -> None:
    # Zero displacement every tick, but always because EVA is being cast --
    # a legitimate animation lock, not stagnation.
    controller = RecoveryController(_config())
    for tick in range(40):
        steering, event = controller.step(
            tick=tick, player_x=0.0, player_z=0.0, heading=0.0,
            displacement_this_tick=0.0, contact_this_tick=False,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.CAST_EVA),
        )
        assert controller.state == RecoveryState.NORMAL


def test_recovery_exits_once_progress_resumes() -> None:
    controller = RecoveryController(_config())
    tick = 0
    # Drive it into recovery.
    while controller.state == RecoveryState.NORMAL and tick < 30:
        controller.step(
            tick=tick, player_x=0.0, player_z=0.0, heading=0.0,
            displacement_this_tick=0.0, contact_this_tick=True,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.NONE),
        )
        tick += 1
    assert controller.state == RecoveryState.RECOVERING

    # Now supply real displacement during the escape -- it should exit back
    # to NORMAL rather than exhausting all attempts.
    for _ in range(20):
        tick += 1
        controller.step(
            tick=tick, player_x=float(tick), player_z=0.0, heading=0.0,
            displacement_this_tick=2.0, contact_this_tick=False,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.NONE),
        )
        if controller.state == RecoveryState.NORMAL:
            break

    assert controller.state == RecoveryState.NORMAL
    assert len(controller.interventions) == 1
    assert controller.interventions[0].outcome == "recovered"


def test_gives_up_after_max_attempts_and_stops_intervening() -> None:
    controller = RecoveryController(_config())
    tick = 0
    while controller.state == RecoveryState.NORMAL and tick < 30:
        controller.step(
            tick=tick, player_x=0.0, player_z=0.0, heading=0.0,
            displacement_this_tick=0.0, contact_this_tick=True,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.NONE),
        )
        tick += 1
    assert controller.state == RecoveryState.RECOVERING

    # Never supply progress -- every escape attempt fails, so the controller
    # must eventually give up rather than intervene forever.
    for _ in range(60):
        tick += 1
        steering, event = controller.step(
            tick=tick, player_x=0.0, player_z=0.0, heading=0.0,
            displacement_this_tick=0.0, contact_this_tick=True,
            map_model=_FakeMap(), policy_steering=int(SteeringAction.STRAIGHT), policy_event=int(FarmingEvent.NONE),
        )
        if controller.state == RecoveryState.GIVEN_UP:
            break

    assert controller.state == RecoveryState.GIVEN_UP
    assert len(controller.interventions) == 1
    assert controller.interventions[0].outcome == "gave_up"
    # Once given up, the policy's own action must be returned unmodified.
    steering, event = controller.step(
        tick=tick + 1, player_x=0.0, player_z=0.0, heading=0.0,
        displacement_this_tick=0.0, contact_this_tick=True,
        map_model=_FakeMap(), policy_steering=int(SteeringAction.RIGHT), policy_event=int(FarmingEvent.CAST_EVA),
    )
    assert steering == int(SteeringAction.RIGHT)
    assert event == int(FarmingEvent.CAST_EVA)

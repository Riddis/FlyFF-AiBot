from __future__ import annotations

from threading import Event, Thread

from runtime_bus import RuntimeBus


def test_latest_values_replace_older_values() -> None:
    bus = RuntimeBus()
    bus.publish_latest("preview", "first")
    bus.publish_latest("preview", "second")

    version, value = bus.read_latest("preview")

    assert version == 2
    assert value == "second"


def test_log_overflow_is_bounded_and_counted() -> None:
    bus = RuntimeBus(max_logs=2)
    bus.log("one")
    bus.log("two")
    bus.log("three")

    assert bus.drain_logs() == [("msg", "two"), ("msg", "three")]
    assert bus.dropped_logs == 1


def test_all_completions_are_delivered() -> None:
    bus = RuntimeBus()
    for index in range(100):
        bus.complete(f"worker-{index}", index)

    completions = bus.drain_completions()

    assert len(completions) == 100
    assert completions[-1].result == 99


def test_confirmation_can_be_cancelled_without_waiting_for_timeout() -> None:
    bus = RuntimeBus()
    cancelled = Event()
    result: list[bool | None] = []

    thread = Thread(
        target=lambda: result.append(
            bus.request_heading_confirmation(
                object(),
                90.0,
                0.9,
                "test",
                cancellation_event=cancelled,
            )
        )
    )
    thread.start()
    cancelled.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [None]

from __future__ import annotations

from threading import Thread
from time import sleep

from runtime_bus import RuntimeBus


def test_mapper_recovery_request_roundtrip() -> None:
    bus = RuntimeBus()
    result: list[str | None] = []

    def request() -> None:
        result.append(
            bus.request_mapper_recovery(
                map_name="Tower AoE",
                reason="heading lost",
                can_retry_in_place=True,
                requires_spawn_reset=False,
                timeout=2.0,
            )
        )

    worker = Thread(target=request)
    worker.start()
    request_item = None
    for _ in range(50):
        request_item = bus.pop_mapper_recovery()
        if request_item is not None:
            break
        sleep(0.01)

    assert request_item is not None
    assert request_item.map_name == "Tower AoE"
    assert request_item.can_retry_in_place
    bus.resolve_mapper_recovery(request_item, "retry")
    worker.join(timeout=1.0)

    assert result == ["retry"]

from __future__ import annotations

import json
from pathlib import Path

import position.attachment_factory as attachment_factory


class FakeService:
    def __init__(self) -> None:
        self.memory = object()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FakeProvider:
    def __init__(self, service: FakeService) -> None:
        self.service = service
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _write_configs(
    tmp_path: Path,
    *,
    position_enabled: bool,
    monster_enabled: bool,
) -> tuple[Path, Path]:
    position_path = tmp_path / "native_position.json"
    position_path.write_text(
        json.dumps(
            {
                "enabled": position_enabled,
                "resolver": "module_pointer",
                "module_name": "Neuz.exe",
                "pointer_offset": "0x5852B8",
            }
        ),
        encoding="utf-8",
    )
    monster_path = tmp_path / "native_monsters.json"
    monster_path.write_text(
        json.dumps({"enabled": monster_enabled}),
        encoding="utf-8",
    )
    return position_path, monster_path


def test_attachment_factory_injects_one_service_and_closes_it_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    position_path, monster_path = _write_configs(
        tmp_path,
        position_enabled=True,
        monster_enabled=True,
    )
    service = FakeService()
    service_calls: list[tuple[object, ...]] = []

    def create_service(*args, **kwargs):
        service_calls.append((args, kwargs))
        return service

    monkeypatch.setattr(
        attachment_factory.NativeProcessService,
        "from_window_handle",
        create_service,
    )
    monkeypatch.setattr(
        attachment_factory.NativeFlyffPositionProvider,
        "from_native_service",
        lambda injected, *_args, **_kwargs: FakeProvider(injected),
    )
    monkeypatch.setattr(
        attachment_factory.NativeFlyffMonsterProvider,
        "from_native_service",
        lambda injected, *_args, **_kwargs: FakeProvider(injected),
    )

    attachment = attachment_factory.create_native_provider_attachment(
        123,
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    assert len(service_calls) == 1
    assert attachment.position_provider is not None
    assert attachment.monster_provider is not None
    assert attachment.position_provider.service is service
    assert attachment.monster_provider.service is service

    attachment.close()
    attachment.close()

    assert attachment.position_provider.close_calls == 1
    assert attachment.monster_provider.close_calls == 1
    assert service.close_calls == 1


def test_attachment_factory_does_not_open_process_when_both_are_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    position_path, monster_path = _write_configs(
        tmp_path,
        position_enabled=False,
        monster_enabled=False,
    )
    monkeypatch.setattr(
        attachment_factory.NativeProcessService,
        "from_window_handle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled attachment must not open a process")
        ),
    )

    attachment = attachment_factory.create_native_provider_attachment(
        123,
        position_config_path=position_path,
        monster_config_path=monster_path,
    )

    assert attachment == attachment_factory.NativeProviderAttachment(
        None,
        None,
        None,
    )


def test_attachment_factory_closes_owner_if_second_provider_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    position_path, monster_path = _write_configs(
        tmp_path,
        position_enabled=True,
        monster_enabled=True,
    )
    service = FakeService()
    position_provider = FakeProvider(service)
    monkeypatch.setattr(
        attachment_factory.NativeProcessService,
        "from_window_handle",
        lambda *_args, **_kwargs: service,
    )
    monkeypatch.setattr(
        attachment_factory.NativeFlyffPositionProvider,
        "from_native_service",
        lambda *_args, **_kwargs: position_provider,
    )
    monkeypatch.setattr(
        attachment_factory.NativeFlyffMonsterProvider,
        "from_native_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    try:
        attachment_factory.create_native_provider_attachment(
            123,
            position_config_path=position_path,
            monster_config_path=monster_path,
        )
    except RuntimeError as error:
        assert str(error) == "boom"
    else:  # pragma: no cover - failure path is the behavior under test.
        raise AssertionError("expected provider construction to fail")

    assert position_provider.close_calls == 1
    assert service.close_calls == 1

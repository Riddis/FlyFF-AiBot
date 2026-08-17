from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from time import sleep

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from position import (  # noqa: E402
    DEFAULT_POSITION_CONFIG_PATH,
    NativeFlyffPositionProvider,
    NativePositionConfig,
    Win32ProcessMemory,
    load_native_position_config,
)


def _integer(value: str) -> int:
    try:
        result = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a decimal integer or 0x-prefixed hexadecimal value"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def _non_negative_integer(value: str) -> int:
    try:
        result = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a decimal integer or 0x-prefixed hexadecimal value"
        ) from error
    if result < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the FlyFF player transform using the project's read-only "
            "native position provider. Module pointers and module offsets are "
            "resolved automatically after every client restart."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=_integer, help="FlyFF process ID")
    target.add_argument(
        "--window-handle",
        type=_integer,
        help="FlyFF HWND in decimal or 0x-prefixed hexadecimal",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_POSITION_CONFIG_PATH,
        help="position config JSON to use",
    )
    parser.add_argument(
        "--address",
        type=_integer,
        help=(
            "optional direct transform address; overrides the module-offset "
            "resolver for one diagnostic run"
        ),
    )
    parser.add_argument("--x-offset", type=_non_negative_integer)
    parser.add_argument("--y-offset", type=_non_negative_integer)
    parser.add_argument("--z-offset", type=_non_negative_integer)
    parser.add_argument("--heading-offset", type=_non_negative_integer)
    parser.add_argument(
        "--heading-unit",
        choices=("degrees", "radians"),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="seconds between samples",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="number of samples; zero means run until Ctrl+C",
    )
    return parser


def _resolved_config(
    args: argparse.Namespace,
    loaded: NativePositionConfig,
) -> NativePositionConfig:
    changes: dict[str, object] = {"enabled": True}
    if args.address is not None:
        changes.update(
            {
                "resolver": "direct_address",
                "transform_address": args.address,
                "module_name": None,
                "transform_offsets": (),
                "pointer_offset": None,
                "minimum_consensus_sources": 1,
            }
        )
    elif loaded.resolver == "direct_address" and loaded.transform_address is None:
        raise SystemExit(
            "No transform source configured. Edit the native resolver in "
            f"{args.config}, or pass --address 0x...."
        )

    for field_name in ("x_offset", "y_offset", "z_offset", "heading_offset"):
        value = getattr(args, field_name)
        if value is not None:
            changes[field_name] = value
    if args.heading_unit is not None:
        changes["heading_unit"] = args.heading_unit
    return replace(loaded, **changes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval < 0.0:
        raise SystemExit("--interval cannot be negative")
    if args.samples < 0:
        raise SystemExit("--samples cannot be negative")

    config = _resolved_config(args, load_native_position_config(args.config))
    if args.pid is not None:
        memory = Win32ProcessMemory(args.pid)
    else:
        memory = Win32ProcessMemory.from_window_handle(args.window_handle)

    provider = NativeFlyffPositionProvider(memory, config)
    addresses = ", ".join(f"0x{value:X}" for value in provider.resolved_addresses)
    if provider.pointer_storage_address is not None:
        print(
            f"resolver={config.resolver} module={config.module_name} "
            f"base=0x{provider.module_base:X} "
            f"pointer_slot=0x{provider.pointer_storage_address:X}",
            flush=True,
        )
    elif provider.module_base is not None:
        print(
            f"resolver={config.resolver} module={config.module_name} "
            f"base=0x{provider.module_base:X} addresses={addresses}",
            flush=True,
        )
    else:
        print(f"resolver={config.resolver} addresses={addresses}", flush=True)
    print("timestamp,x,y,z,heading_degrees,consensus_sources,max_delta", flush=True)

    emitted = 0
    try:
        while args.samples == 0 or emitted < args.samples:
            pose = provider.read_pose()
            diagnostics = provider.last_diagnostics
            heading = "" if pose.heading_degrees is None else f"{pose.heading_degrees:.6f}"
            source_count = (
                len(diagnostics.consensus_addresses) if diagnostics is not None else 0
            )
            max_delta = (
                ""
                if diagnostics is None or diagnostics.maximum_consensus_delta is None
                else f"{diagnostics.maximum_consensus_delta:.9f}"
            )
            print(
                f"{pose.timestamp:.6f},{pose.x:.6f},{pose.y:.6f},"
                f"{pose.z:.6f},{heading},{source_count},{max_delta}",
                flush=True,
            )
            emitted += 1
            if args.samples == 0 or emitted < args.samples:
                sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

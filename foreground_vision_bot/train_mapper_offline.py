from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mapper.rl.OfflineTraining import (
    config_to_dict,
    evaluate_saved_policy,
    load_offline_config,
    run_smoke_simulations,
    train_offline,
    validate_curriculum,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the offline FlyFF movement/exploration policy "
            "on the completed real map plus generated open arenas."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to offline_training.json (defaults to mapper/rl/offline_training.json).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate the real map and generate curriculum samples.",
    )
    validate.add_argument("--samples", type=int, default=25)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run game-free heuristic simulation episodes without RL dependencies.",
    )
    smoke.add_argument("--episodes", type=int, default=20)
    smoke.add_argument("--max-steps", type=int, default=None)

    train = subparsers.add_parser(
        "train",
        help="Train MaskablePPO offline.",
    )
    train.add_argument("--timesteps", type=int, default=None)
    train.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model on the untouched real map.",
    )
    evaluate.add_argument("--episodes", type=int, default=None)
    evaluate.add_argument("--model", type=Path, default=None)

    subparsers.add_parser(
        "show-config",
        help="Print the resolved training configuration.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_offline_config(args.config)

        if args.command == "validate":
            payload = validate_curriculum(config, samples=args.samples)
        elif args.command == "smoke":
            payload = run_smoke_simulations(
                config,
                episodes=args.episodes,
                max_steps=args.max_steps,
            ).to_dict()
        elif args.command == "train":
            path = train_offline(
                config,
                total_timesteps=args.timesteps,
                resume=args.resume,
            )
            payload = {"model": str(path)}
        elif args.command == "evaluate":
            payload = evaluate_saved_policy(
                config,
                model_path=args.model,
                episodes=args.episodes,
            ).to_dict()
        elif args.command == "show-config":
            payload = config_to_dict(config)
        else:  # pragma: no cover - argparse guarantees a known command.
            raise RuntimeError(f"unknown command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

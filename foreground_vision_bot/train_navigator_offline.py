from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mapper.rl.NavigatorTraining import (
    config_to_dict,
    evaluate_saved_navigator,
    load_navigator_config,
    run_navigator_smoke,
    train_navigator,
    validate_navigator_curriculum,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the goal-conditioned FlyFF movement navigator. "
            "This is separate from the mapping/exploration policy."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to navigator_training.json.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate maps and sampled start-goal tasks.")
    validate.add_argument("--samples", type=int, default=25)
    smoke = sub.add_parser("smoke", help="Run shortest-path heuristic simulations.")
    smoke.add_argument("--episodes", type=int, default=25)
    smoke.add_argument("--max-steps", type=int, default=None)
    train = sub.add_parser("train", help="Train the navigator with MaskablePPO.")
    train.add_argument("--timesteps", type=int, default=None)
    train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible navigator with the lower precision-fine-tuning "
            "learning rate and deterministic best-checkpoint selection."
        ),
    )
    evaluate = sub.add_parser("evaluate", help="Evaluate a saved navigator on Tower AoE.")
    evaluate.add_argument("--episodes", type=int, default=None)
    evaluate.add_argument("--model", type=Path, default=None)
    sub.add_parser("show-config", help="Print the resolved navigator configuration.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_navigator_config(args.config)
        if args.command == "validate":
            payload = validate_navigator_curriculum(config, samples=args.samples)
        elif args.command == "smoke":
            payload = run_navigator_smoke(
                config,
                episodes=args.episodes,
                max_steps=args.max_steps,
            ).to_dict()
        elif args.command == "train":
            payload = {
                "model": str(
                    train_navigator(
                        config,
                        total_timesteps=args.timesteps,
                        resume=args.resume,
                    )
                )
            }
        elif args.command == "evaluate":
            payload = evaluate_saved_navigator(
                config,
                model_path=args.model,
                episodes=args.episodes,
            ).to_dict()
        else:
            payload = config_to_dict(config)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

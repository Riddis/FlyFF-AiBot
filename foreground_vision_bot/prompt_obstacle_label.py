from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path


VALID = {"clear", "blocked", "ignore", "skip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a highlighted forward corridor.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--motion-label", default="unknown")
    parser.add_argument("--suggested-label", choices=sorted(VALID), default="ignore")
    parser.add_argument("--reason", default="")
    parser.add_argument("--floor-risk", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {"value": "skip"}
    root = tk.Tk()
    root.title("Obstacle Active Learning")
    root.attributes("-topmost", True)
    root.protocol("WM_DELETE_WINDOW", lambda: finish("skip"))

    def finish(value: str) -> None:
        result["value"] = value if value in VALID else "skip"
        root.destroy()

    title = tk.Label(
        root,
        text="Is the highlighted corridor ordinary walkable floor?",
        font=("Segoe UI", 13, "bold"),
        padx=12,
        pady=8,
    )
    title.pack()
    explanation = tk.Label(
        root,
        text=(
            "Classify only the green/yellow forward corridor. Ignore mobs, loot, "
            "spell effects, or camera clipping when the floor cannot be judged."
        ),
        wraplength=700,
        justify="left",
        padx=12,
    )
    explanation.pack()

    try:
        image = tk.PhotoImage(file=str(args.image))
    except tk.TclError as error:
        print(f"Could not open review image: {error}", file=sys.stderr)
        root.destroy()
        print("skip")
        return
    image_label = tk.Label(root, image=image, padx=8, pady=8)
    image_label.image = image
    image_label.pack()

    risk_text = "not ready" if args.floor_risk is None else f"{args.floor_risk:.2f}"
    details = tk.Label(
        root,
        text=(
            f"Motion label: {args.motion_label}   |   floor-model risk: {risk_text}\n"
            f"Reason: {args.reason}"
        ),
        wraplength=700,
        justify="left",
        fg="#444444",
        padx=12,
        pady=6,
    )
    details.pack()

    buttons = tk.Frame(root, padx=12, pady=12)
    buttons.pack()
    specs = [
        ("Floor / Safe", "clear", "#26734d"),
        ("Not Floor / Blocker", "blocked", "#9b2c2c"),
        ("Ignore / Can't Tell", "ignore", "#555555"),
        ("Skip", "skip", "#777777"),
    ]
    for text, value, colour in specs:
        button = tk.Button(
            buttons,
            text=text,
            command=lambda selected=value: finish(selected),
            bg=colour,
            fg="white",
            activeforeground="white",
            width=20,
            padx=4,
            pady=7,
        )
        button.pack(side="left", padx=4)
        if value == args.suggested_label:
            button.focus_set()

    root.bind("<Escape>", lambda _event: finish("skip"))
    root.mainloop()
    print(result["value"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .config import RecorderConfig
from .keyboard import SUPPORTED_EVA_HOTKEYS
from .session import RecorderController
from .windows import ClientWindow, find_client_windows


class RecorderGui:
    def __init__(self, root: tk.Tk, config: RecorderConfig | None = None) -> None:
        self.root = root
        self.config = config or RecorderConfig.load()
        self.controller = RecorderController(self.config)
        self.clients: dict[str, ClientWindow] = {}
        self.attaching = False
        self.last_output_zip: Path | None = None

        root.title("FlyFF Farming Session Recorder")
        root.minsize(760, 560)
        root.geometry("820x620")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)

        ttk.Label(
            outer,
            text="FlyFF Farming Session Recorder",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            outer,
            text=(
                "Read-only recorder for player movement, farming inputs, monster positions, "
                "deaths, appearances, and actor loading."
            ),
            wraplength=760,
        ).grid(row=1, column=0, sticky="ew", pady=(2, 12))

        setup = ttk.LabelFrame(outer, text="Client setup", padding=10)
        setup.grid(row=2, column=0, sticky="ew")
        setup.columnconfigure(1, weight=1)

        ttk.Label(setup, text="FlyFF client:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.client_var = tk.StringVar()
        self.client_combo = ttk.Combobox(
            setup, textvariable=self.client_var, state="readonly", width=55
        )
        self.client_combo.grid(row=0, column=1, sticky="ew")
        self.refresh_button = ttk.Button(setup, text="Refresh", command=self._refresh_clients)
        self.refresh_button.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(setup, text="Current full HP:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.hp_var = tk.StringVar()
        self.hp_entry = ttk.Entry(setup, textvariable=self.hp_var, width=18)
        self.hp_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(
            setup,
            text="Stand at the map spawn with full HP before attaching.",
        ).grid(row=1, column=1, sticky="e", pady=(8, 0))

        keyboard_row = ttk.Frame(setup)
        keyboard_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(keyboard_row, text="Keyboard:").pack(side="left")
        self.layout_var = tk.StringVar(value="qwerty")
        self.layout_combo = ttk.Combobox(
            keyboard_row,
            textvariable=self.layout_var,
            values=("azerty", "qwerty"),
            state="readonly",
            width=10,
        )
        self.layout_combo.pack(side="left", padx=(8, 18))
        ttk.Label(keyboard_row, text="EVA hotkey:").pack(side="left")
        self.eva_var = tk.StringVar(value="F1")
        self.eva_combo = ttk.Combobox(
            keyboard_row,
            textvariable=self.eva_var,
            values=SUPPORTED_EVA_HOTKEYS,
            state="readonly",
            width=6,
        )
        self.eva_combo.pack(side="left", padx=(8, 18))
        self.attach_button = ttk.Button(
            keyboard_row, text="Attach Client", command=self._attach_client
        )
        self.attach_button.pack(side="right")

        ttk.Label(
            setup,
            text=(
                "Number choices 1–9 mean the physical top-row number keys on both layouts. "
                "For example, choose 2 for the é key on AZERTY."
            ),
            wraplength=730,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        workflow = ttk.LabelFrame(outer, text="Recording", padding=10)
        workflow.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        workflow.columnconfigure(0, weight=1)
        self.instruction_var = tk.StringVar(
            value=(
                "Step 1 — Select the client, enter your full HP, choose the EVA key, "
                "and click Attach Client."
            )
        )
        ttk.Label(
            workflow,
            textvariable=self.instruction_var,
            wraplength=760,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        button_row = ttk.Frame(workflow)
        button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        button_row.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            button_row, text="Start Logging", command=self._start_logging, state="disabled"
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.end_button = ttk.Button(
            button_row, text="End Logging", command=self._end_logging, state="disabled"
        )
        self.end_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        stats = ttk.LabelFrame(outer, text="Live session", padding=10)
        stats.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.stats_var = tk.StringVar(value="Not recording.")
        ttk.Label(stats, textvariable=self.stats_var, wraplength=760).pack(anchor="w")

        log_frame = ttk.LabelFrame(outer, text="Status", padding=8)
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.open_output_button = ttk.Button(
            outer, text="Open Output Folder", command=self._open_output_folder
        )
        self.open_output_button.grid(row=6, column=0, sticky="e", pady=(10, 0))

        self._refresh_clients()
        self._poll_events()

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_clients(self) -> None:
        try:
            windows = find_client_windows(self.config.window_title_prefix)
        except Exception as error:
            messagebox.showerror("Client scan failed", str(error))
            return
        self.clients = {item.title: item for item in windows}
        titles = tuple(self.clients)
        self.client_combo.configure(values=titles)
        if titles:
            self.client_var.set(titles[0])
            self._append_log(f"Found {len(titles)} FlyFF client window(s).")
        else:
            self.client_var.set("")
            self._append_log("No visible FlyFF client was found.")

    def _attach_client(self) -> None:
        if self.attaching:
            self.controller.cancel_attachment()
            self._append_log("Cancelling attachment…")
            return
        selected = self.clients.get(self.client_var.get())
        if selected is None:
            messagebox.showerror("No client selected", "Refresh and select a FlyFF client first.")
            return
        try:
            hp = int(self.hp_var.get().strip())
            if hp <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid HP", "Enter your current full HP as a whole number.")
            return
        self.attaching = True
        self.attach_button.configure(text="Cancel Attach")
        self._set_setup_enabled(False, keep_attach=True)
        self.start_button.configure(state="disabled")
        self.instruction_var.set(
            "Attaching read-only and discovering pointers. Keep the character at spawn with full HP."
        )
        self._append_log("Starting native attachment and pointer discovery…")
        try:
            self.controller.attach_async(
                hwnd=selected.hwnd,
                title=selected.title,
                player_full_hp=hp,
                keyboard_layout=self.layout_var.get(),
                eva_hotkey=self.eva_var.get(),
            )
        except Exception as error:
            self.attaching = False
            self.attach_button.configure(text="Attach Client")
            self._set_setup_enabled(True)
            messagebox.showerror("Attachment failed", str(error))

    def _set_setup_enabled(self, enabled: bool, *, keep_attach: bool = False) -> None:
        state = "readonly" if enabled else "disabled"
        self.client_combo.configure(state=state)
        self.refresh_button.configure(state="normal" if enabled else "disabled")
        self.hp_entry.configure(state="normal" if enabled else "disabled")
        self.layout_combo.configure(state="readonly" if enabled else "disabled")
        self.eva_combo.configure(state="readonly" if enabled else "disabled")
        if not keep_attach:
            self.attach_button.configure(state="normal" if enabled else "disabled")

    def _start_logging(self) -> None:
        try:
            self.controller.start_logging()
        except Exception as error:
            messagebox.showerror("Could not start", str(error))
            return
        self.start_button.configure(state="disabled")
        self.end_button.configure(state="normal")
        self.attach_button.configure(state="disabled")
        self.instruction_var.set(
            "Step 2 — Farm normally. Click End Logging when finished. Leaving the farming map "
            "also ends and packages the recording automatically."
        )

    def _end_logging(self) -> None:
        try:
            self.controller.end_logging()
        except Exception as error:
            messagebox.showerror("Could not stop", str(error))
            return
        self.end_button.configure(state="disabled")
        self.instruction_var.set("Finalizing and compressing the recording. Please wait…")

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.controller.events.get_nowait()
            except queue.Empty:
                break
            event_type = event.get("type")
            if event_type == "status":
                self._append_log(str(event.get("message", "")))
            elif event_type == "attach_cancelled":
                self.attaching = False
                self.attach_button.configure(text="Attach Client", state="normal")
                self._set_setup_enabled(True)
                self.instruction_var.set(
                    "Step 1 — Select the client, enter full HP, and click Attach Client."
                )
                self._append_log(str(event.get("message", "Attachment cancelled.")))
            elif event_type == "attached":
                self.attaching = False
                self.attach_button.configure(text="Attach Client", state="normal")
                self.start_button.configure(state="normal")
                self.instruction_var.set(
                    "Step 2 — Click Start Logging, then farm normally until the session is finished."
                )
                self._append_log(
                    f"Attached to {event['title']} (PID {event['pid']}); "
                    f"{event['actor_slots']} actor slots currently cached."
                )
            elif event_type == "logging_started":
                self._append_log("Recording files opened successfully. Farming data is now being logged.")
            elif event_type == "stats":
                stats = event["stats"]
                self.stats_var.set(
                    f"Elapsed: {float(stats['elapsed_seconds']):.1f}s | living monsters: "
                    f"{stats['living_monsters']} | actor slots: {stats['cached_actor_slots']} | "
                    f"active reads: {stats['hot_actor_slots']} | cold slots: "
                    f"{stats['cold_actor_slots']} | completed rescans: "
                    f"{stats['rediscoveries_completed']} | probable kills: "
                    f"{stats['probable_kills']} | target appearances: "
                    f"{stats['target_appearances']} | frames: {stats['frames']} | client focused: "
                    f"{'yes' if stats['focused'] else 'no'}"
                )
            elif event_type == "finished":
                self.last_output_zip = Path(str(event["output_zip"]))
                self._append_log(f"Recording package created: {self.last_output_zip}")
                self.stats_var.set(
                    f"Finished. Probable kills: {event['probable_kills']} | "
                    f"respawn candidates: {event['respawn_candidates']} | "
                    f"target appearances: {event['target_appearances']}"
                )
                finished_status = str(event.get("status", "success"))
                self.instruction_var.set(
                    "Done — send the generated SEND_TO_RIDDIMS ZIP to Riddims."
                    if finished_status == "success"
                    else "The diagnostic ZIP was created, but recording ended with an error. Send it to Riddims."
                )
                self.end_button.configure(state="disabled")
                self.start_button.configure(state="normal")
                messagebox.showinfo(
                    "Recording complete",
                    f"The ZIP is ready:\n\n{self.last_output_zip}",
                )
            elif event_type == "error":
                self._append_log(
                    f"ERROR ({event.get('context', 'unknown')}): {event.get('message', '')}"
                )
                if event.get("context") == "attach":
                    self.attaching = False
                    self.attach_button.configure(text="Attach Client", state="normal")
                    self._set_setup_enabled(True)
                messagebox.showerror("Recorder error", str(event.get("message", "Unknown error")))
        self.root.after(100, self._poll_events)

    def _open_output_folder(self) -> None:
        folder = Path.home() / "Documents" / self.config.output_directory_name
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(folder)  # type: ignore[attr-defined]
        except Exception as error:
            messagebox.showerror("Could not open folder", str(error))

    def _on_close(self) -> None:
        if self.controller.logging:
            if not messagebox.askyesno(
                "Recording active",
                "End and package the current recording before closing?",
            ):
                return
            self.controller.end_logging()
            self.instruction_var.set("Finalizing before close. The window will stay open until done.")
            return
        self.controller.close()
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    RecorderGui(root)
    root.mainloop()

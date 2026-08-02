import difflib
import math
import re

try:
    import winsound
except ImportError:  # pragma: no cover - Windows production dependency.
    winsound = None
from time import monotonic

import cv2 as cv
import PySimpleGUI as sg
from mapper.ManualMapEditor import ManualMapEditorSession
from mapper.MapCatalog import MapCatalog
from mapper.OccupancyGrid import OccupancyGrid
from runtime_bus import RuntimeAlert, RuntimeBus, RuntimeStatus
from runtime_controller import RuntimeController
from utils.helpers import get_window_handlers, hex_variant


class Gui:
    def __init__(self, theme="DarkAmber"):
        self.logger_events = ["msg", "msg_red", "msg_purple", "msg_blue", "msg_green"]
        self.logger_events_color = {
            "msg": ("white", "black"),
            "msg_red": ("white", "red"),
            "msg_purple": ("white", "purple"),
            "msg_blue": ("white", "blue"),
            "msg_green": ("white", "green"),
        }
        self.frame_resolutions = {
            "Fit panel": None,
            "160x120": (160, 120),
            "200x150": (200, 150),
            "320x240": (320, 240),
            "400x300": (400, 300),
            "640x480": (640, 480),
            "800x450": (800, 450),
            "800x600": (800, 600),
            "960x540": (960, 540),
            "1024x600": (1024, 600),
            "1024x768": (1024, 768),
            "1280x700": (1280, 700),
            "1280x720": (1280, 720),
            "1280x800": (1280, 800),
            "1280x1024": (1280, 1024),
            "1366x768": (1366, 768),
        }
        self.controller = None
        self._log_window = None
        self._heading_overlay_detector = None
        self._status_mode = "Idle"
        self._last_status_message = "Ready"
        self.runtime_bus = RuntimeBus(max_logs=1500)
        self.map_catalog = MapCatalog()
        self.map_names = list(self.map_catalog.names())
        self._selected_map_name = self.map_catalog.default_name
        self._last_versions = {
            "debug_frame": 0,
            "map_frame": 0,
            "video_fps": 0,
            "rl_status": 0,
            "mapper_status": 0,
            "capture_status": 0,
            "runtime_status": 0,
            "native_diagnostic_status": 0,
        }
        self._last_preview_render_at = 0.0
        self._last_map_render_at = 0.0
        self._last_log_render_at = 0.0
        self._preview_render_interval = 1.0 / 10.0
        self._map_render_interval = 1.0 / 4.0
        sg.theme(theme)

    def init(self):
        layout = self.__get_layout()
        self.window = sg.Window(
            "Flyff FVF",
            layout,
            location=(0, 0),
            size=(1320, 990),
            resizable=True,
            finalize=True,
        )
        sg.cprint_set_output_destination(self.window, "-ML-")
        sg.user_settings_filename(path=".")
        self.__set_hotkeys()
        return self.window

    def loop(self, bot):
        self.controller = RuntimeController(bot, self.runtime_bus)
        self.__load_settings(bot)
        while True:
            event, values = self.window.read(timeout=50)
            self.__service_log_window()
            self.__refresh_runtime(values)

            # ACTIONS - Button events
            if event == "Exit" or event == sg.WIN_CLOSED:
                if self.__shutdown(bot):
                    break
                continue
            if event == "-ATTACH_WINDOW-":
                game_window_name, game_window_handler = self.__attach_window_popup()
                if game_window_name and game_window_handler:
                    try:
                        self.controller.attach(game_window_handler)
                    except Exception as error:  # noqa: BLE001 - GUI command boundary.
                        message = f"Could not attach to the Flyff window: {error}"
                        self.__set_status("Attach failed", message)
                        self.runtime_bus.log(message, "msg_red")
                        self.show_error(message)
                    else:
                        truncated_game_window_name = (
                            game_window_name[:30] + "..."
                            if len(game_window_name) > 30
                            else game_window_name
                        )
                        self.window["-ATTACHED_WINDOW-"].update(
                            truncated_game_window_name
                        )
                        self.__set_rl_buttons(attached=True, running=False)

            if event == "-START_BOT-":
                self.__start_control(
                    lambda: self.controller.start_rl("train"),
                    "Training",
                    "Starting training...",
                )

            if event == "-DRY_RUN-":
                self.__start_control(
                    lambda: self.controller.start_rl("dry-run"),
                    "Dry Run",
                    "Starting native no-learning dry run...",
                )

            if event == "-VALIDATE_DATA-":
                self.__start_control(
                    lambda: self.controller.start_rl("validate-data"),
                    "Data Validation",
                    "Starting training-data validation run...",
                )

            if event == "-RUN_AGENT-":
                self.__start_control(
                    lambda: self.controller.start_rl("agent"),
                    "Agent",
                    "Starting trained agent...",
                )

            if event == "-START_MAPPER-":
                selected_map = values.get("-MAP-NAME-", self._selected_map_name)
                self.__apply_map_selection(bot, selected_map, publish_preview=False)
                self.__start_control(
                    lambda: self.controller.start_mapper(
                        self._selected_map_name,
                        rl_shadow_enabled=False,
                    ),
                    "Mapping",
                    f"Starting coordinate mapper for {self._selected_map_name}...",
                )

            if event == "-START_MANUAL_MAPPER-":
                selected_map = values.get("-MAP-NAME-", self._selected_map_name)
                self.__apply_map_selection(bot, selected_map, publish_preview=False)
                self.__start_control(
                    lambda: self.controller.start_manual_mapper(
                        self._selected_map_name,
                    ),
                    "Manual Mapping",
                    f"Tracking your movement on {self._selected_map_name}; "
                    "the bot will not send movement keys.",
                )

            if event == "-SET_MINIMAP_ANCHOR-":
                self.__set_minimap_anchor(bot)

            if event == "-EDIT_MAP_CELLS-":
                self.__edit_map_cells_popup()

            if event == "-SHOW_LOG-":
                self.__show_log_window()

            if event == "-STOP_BOT-":
                self.__set_status("Idle", "Stopped")
                self.controller.stop_control()
                self.controller.stop_native_diagnostic()
                self.runtime_bus.log(
                    "Stop requested. Waiting for the active worker to exit.",
                    "msg_blue",
                )

            if event in {"-NATIVE_HEALTH-", "-RECOVER_POINTERS-"}:
                recover = event == "-RECOVER_POINTERS-"
                try:
                    self.controller.start_native_diagnostic(
                        recover=recover,
                        timeout=(
                            self.controller.POINTER_RECOVERY_TIMEOUT_SECONDS
                            if recover
                            else 1.0
                        ),
                    )
                except (RuntimeError, ValueError) as error:
                    self.runtime_bus.log(str(error), "msg_red")
                else:
                    mode = "Native Recovery" if recover else "Native Health"
                    self.__set_status(mode, f"Starting {mode.lower()}...")
                    if recover:
                        self.__set_rl_buttons(attached=True, running=True)

            # BOT OPTIONS - Video options
            if event == "-SHOW_FRAMES-":
                bot.set_config(show_frames=values["-SHOW_FRAMES-"])
                sg.user_settings_set_entry("-SHOW_FRAMES-", values["-SHOW_FRAMES-"])
                self.window["-SHOW_MATCHES_TEXT-"].update(
                    visible=(values["-SHOW_FRAMES-"])
                )
                self.window["-SHOW_BOXES-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window["-SHOW_MARKERS-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window["-VISION_FRAME-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window.refresh()  # Combined with contents_changed, will compute the new size of the element
                self.window["-MAIN_COLUMN-"].contents_changed()
            if event == "-SHOW_MATCHES_TEXT-":
                bot.set_config(show_matches_text=values["-SHOW_MATCHES_TEXT-"])
                sg.user_settings_set_entry(
                    "-SHOW_MATCHES_TEXT-", values["-SHOW_MATCHES_TEXT-"]
                )
            if event == "-SHOW_BOXES-":
                bot.set_config(show_mobs_pos_boxes=values["-SHOW_BOXES-"])
                sg.user_settings_set_entry("-SHOW_BOXES-", values["-SHOW_BOXES-"])
            if event == "-SHOW_MARKERS-":
                bot.set_config(show_mobs_pos_markers=values["-SHOW_MARKERS-"])
                sg.user_settings_set_entry("-SHOW_MARKERS-", values["-SHOW_MARKERS-"])

            # BOT OPTIONS - Threshold options
            if isinstance(event, str) and event.startswith("-BOT_THRESHOLD_OPTIONS-"):
                self.window["-BOT_THRESHOLD_OPTIONS-"].update(
                    visible=not self.window["-BOT_THRESHOLD_OPTIONS-"].visible
                )
                self.window["-BOT_THRESHOLD_OPTIONS-" + "-BUTTON-"].update(
                    self.window["-BOT_THRESHOLD_OPTIONS-"].metadata[0]
                    if self.window["-BOT_THRESHOLD_OPTIONS-"].visible
                    else self.window["-BOT_THRESHOLD_OPTIONS-"].metadata[1]
                )
                self.window.refresh()  # Combined with contents_changed, will compute the new size of the element
                self.window["-MAIN_COLUMN-"].contents_changed()
            if event == "-MOB_POS_MATCH_THRESHOLD-":
                bot.set_config(
                    mob_pos_match_threshold=values["-MOB_POS_MATCH_THRESHOLD-"]
                )
                sg.user_settings_set_entry(
                    "-MOB_POS_MATCH_THRESHOLD-", values["-MOB_POS_MATCH_THRESHOLD-"]
                )
            if event == "-MOB_STILL_ALIVE_MATCH_THRESHOLD-":
                bot.set_config(
                    mob_still_alive_match_threshold=values[
                        "-MOB_STILL_ALIVE_MATCH_THRESHOLD-"
                    ]
                )
                sg.user_settings_set_entry(
                    "-MOB_STILL_ALIVE_MATCH_THRESHOLD-",
                    values["-MOB_STILL_ALIVE_MATCH_THRESHOLD-"],
                )
            if event == "-MOB_EXISTENCE_MATCH_THRESHOLD-":
                bot.set_config(
                    mob_existence_match_threshold=values[
                        "-MOB_EXISTENCE_MATCH_THRESHOLD-"
                    ]
                )
                sg.user_settings_set_entry(
                    "-MOB_EXISTENCE_MATCH_THRESHOLD-",
                    values["-MOB_EXISTENCE_MATCH_THRESHOLD-"],
                )
            if event == "-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-":
                bot.set_config(
                    inventory_perin_converter_match_threshold=values[
                        "-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-"
                    ]
                )
                sg.user_settings_set_entry(
                    "-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-",
                    values["-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-"],
                )
            if event == "-INVENTORY_ICONS_MATCH_THRESHOLD-":
                bot.set_config(
                    inventory_icons_match_threshold=values[
                        "-INVENTORY_ICONS_MATCH_THRESHOLD-"
                    ]
                )
                sg.user_settings_set_entry(
                    "-INVENTORY_ICONS_MATCH_THRESHOLD-",
                    values["-INVENTORY_ICONS_MATCH_THRESHOLD-"],
                )

            # BOT OPTIONS - General options
            if event == "-MOBS_KILL_GOAL-":
                if values["-MOBS_KILL_GOAL-"].lower() in ["infinite", "inf", "0", ""]:
                    self.window["-MOBS_KILL_GOAL-"].update("infinite")
                    bot.set_config(mobs_kill_goal=None)
                    sg.user_settings_set_entry("-MOBS_KILL_GOAL-", "infinite")
                else:
                    try:
                        mobs_kill_goal = int(values["-MOBS_KILL_GOAL-"])
                        bot.set_config(mobs_kill_goal=mobs_kill_goal)
                        sg.user_settings_set_entry(
                            "-MOBS_KILL_GOAL-", values["-MOBS_KILL_GOAL-"]
                        )
                    except ValueError:
                        sg.cprint("Invalid mobs kill goal")
                        self.window["-MOBS_KILL_GOAL-"].update("infinite")
                        bot.set_config(mobs_kill_goal=None)
                        sg.user_settings_set_entry("-MOBS_KILL_GOAL-", "infinite")
            if event == "-FIGHT_TIME_LIMIT_SEC-":
                try:
                    fight_time_limit_sec = int(values["-FIGHT_TIME_LIMIT_SEC-"])
                    bot.set_config(fight_time_limit_sec=fight_time_limit_sec)
                    sg.user_settings_set_entry(
                        "-FIGHT_TIME_LIMIT_SEC-", values["-FIGHT_TIME_LIMIT_SEC-"]
                    )
                except ValueError:
                    sg.cprint("Invalid fight time limit")
                    self.window["-FIGHT_TIME_LIMIT_SEC-"].update("8")
                    bot.set_config(fight_time_limit_sec=8)
                    sg.user_settings_set_entry("-FIGHT_TIME_LIMIT_SEC-", "8")
            if event == "-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-":
                try:
                    delay_to_check_mob_still_alive_sec = float(
                        values["-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-"]
                    )
                    bot.set_config(
                        delay_to_check_mob_still_alive_sec=delay_to_check_mob_still_alive_sec
                    )
                    sg.user_settings_set_entry(
                        "-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-",
                        values["-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-"],
                    )
                except ValueError:
                    sg.cprint("Invalid delay to check if mob is still alive")
                    self.window["-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-"].update("0.25")
                    bot.set_config(delay_to_check_mob_still_alive_sec=0.25)
                    sg.user_settings_set_entry(
                        "-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-", "0.25"
                    )
            if event == "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-":
                try:
                    convert_penya_to_perins_timer_min = float(
                        values["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"]
                    )
                    bot.set_config(
                        convert_penya_to_perins_timer_min=convert_penya_to_perins_timer_min
                    )
                    sg.user_settings_set_entry(
                        "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-",
                        values["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"],
                    )
                except ValueError:
                    sg.cprint(
                        "Invalid convert penya to perins timer, must be in minutes"
                    )
                    self.window["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"].update("30")
                    bot.set_config(convert_penya_to_perins_timer_min=30)
                    sg.user_settings_set_entry(
                        "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-", "30"
                    )

            if event == "-MAP-NAME-":
                if self.controller.control_active:
                    self.window["-MAP-NAME-"].update(self._selected_map_name)
                    self.runtime_bus.log(
                        "Stop the active control task before changing maps.",
                        "msg_red",
                    )
                else:
                    self.__apply_map_selection(
                        bot,
                        values.get("-MAP-NAME-"),
                        publish_preview=True,
                    )

            if event == "-ADD_MAP-":
                self.__add_map_popup(bot)

            if event == "-EDIT_MAP_MOBS-":
                self.__edit_map_mobs_popup(bot)

            if event == "-RESET_MAP-":
                self.__reset_selected_map(bot)

            if event == "-DELETE_MAP-":
                self.__delete_selected_map(bot)

            # MOBS - Mobs configuration
            if event == "-SELECT_MOBS-":
                self.__select_mobs_popup(bot)

            if event == "-ADD_MOB-":
                self.__add_mob_popup(bot)

            if event == "-DELETE_MOB-":
                self.__select_mobs_popup(bot, is_delete_form=True)

    def __start_control(self, command, mode, message):
        try:
            command()
        except RuntimeError as error:
            self.runtime_bus.log(str(error), "msg_red")
            return
        self.__set_status(mode, message)
        self.__set_rl_buttons(attached=True, running=True)
        self.runtime_bus.log(message, "msg_blue")

    def __refresh_runtime(self, values):
        now = monotonic()

        # High-rate preview: latest frame only, rendered at <=10 FPS.
        version, frame = self.runtime_bus.read_latest(
            "debug_frame",
            self._last_versions["debug_frame"],
        )
        if (
            frame is not None
            and values.get("-SHOW_FRAMES-", False)
            and now - self._last_preview_render_at >= self._preview_render_interval
        ):
            self._last_versions["debug_frame"] = version
            # Preview analysis belongs to the capture/runtime side. Running
            # heading template matching here blocks Tk's only event loop and
            # makes Bot Vision plus every button appear frozen.
            image = frame.copy()
            resolution = values.get("-DEBUG_IMG_WIDTH-", "Fit panel")
            width, height = self.__preview_target_size(resolution)
            image = self.__fit_image(image, width, height)
            # Tk's PhotoImage accepts PNG bytes directly. JPEG bytes trigger
            # PySimpleGUI's internal modal error dialog, which Tk can position
            # far outside the visible desktop on multi-monitor setups.
            encoded = cv.imencode(".png", image)
            if encoded[0]:
                self.window["-DEBUG_IMAGE-"].update(data=encoded[1].tobytes())
            self._last_preview_render_at = now

        # Map is latest-only and deliberately slow.
        version, frame = self.runtime_bus.read_latest(
            "map_frame",
            self._last_versions["map_frame"],
        )
        if (
            frame is not None
            and now - self._last_map_render_at >= self._map_render_interval
        ):
            self._last_versions["map_frame"] = version
            image = self.__fit_image(frame, 960, 245)
            encoded = cv.imencode(".png", image)
            if encoded[0]:
                self.window["-MAPPER_IMAGE-"].update(data=encoded[1].tobytes())
            self._last_map_render_at = now

        version, fps = self.runtime_bus.read_latest(
            "video_fps",
            self._last_versions["video_fps"],
        )
        if fps is not None:
            self._last_versions["video_fps"] = version
            self.window["-VIDEO_FPS-"].update(f"FPS: {fps}")

        for key, mode in (
            ("rl_status", "Training"),
            ("mapper_status", self._status_mode),
            ("runtime_status", "Runtime"),
            ("native_diagnostic_status", "Native Diagnostic"),
        ):
            version, status = self.runtime_bus.read_latest(
                key,
                self._last_versions[key],
            )
            if status is not None:
                self._last_versions[key] = version
                if isinstance(status, RuntimeStatus):
                    if not self.__is_current_status(key, status):
                        continue
                    status = status.message
                self.__set_status(mode, str(status))

        version, capture_status = self.runtime_bus.read_latest(
            "capture_status",
            self._last_versions["capture_status"],
        )
        if capture_status is not None:
            self._last_versions["capture_status"] = version
            if isinstance(capture_status, RuntimeStatus) and self.__is_current_status(
                "capture_status",
                capture_status,
            ):
                if capture_status.message == "degraded":
                    self.__set_status(
                        "Capture",
                        "Capture degraded; retrying the attached window.",
                    )
                elif capture_status.message == "lost":
                    self.__set_status(
                        "Capture lost",
                        "The attached window stopped producing frames.",
                    )

        # Bounded log draining at <=5 FPS.
        if now - self._last_log_render_at >= 0.20:
            for level, message in self.runtime_bus.drain_logs(80):
                color = self.logger_events_color.get(
                    level,
                    ("white", "black"),
                )
                sg.cprint(message, c=color)
                if self._log_window is not None:
                    self._log_window["-LOG-TEXT-"].print(message)
                self._last_status_message = message
            self._last_log_render_at = now

        for completion in self.runtime_bus.drain_completions():
            if completion.worker_name.startswith("capture-"):
                if not self.__is_current_capture_event(completion):
                    continue
                self.__set_rl_buttons(attached=False, running=False)
                self.runtime_bus.log(
                    "Capture stopped; attach the Flyff window again.",
                    "msg_yellow",
                )
                continue
            if completion.worker_name == "preview":
                continue
            if completion.worker_name in {"native-health", "native-pointer-recovery"}:
                if completion.session_id != self.controller.diagnostic_session_id:
                    continue
                if completion.worker_name == "native-pointer-recovery":
                    self.__set_rl_buttons(attached=True, running=False)
                self.runtime_bus.log(
                    f"{completion.worker_name} finished.",
                    "msg_green",
                )
                continue
            if (
                completion.session_id is not None
                and completion.session_id != self.controller.control_session_id
            ):
                continue
            self.__set_rl_buttons(attached=True, running=False)
            self.runtime_bus.log(
                f"{completion.worker_name} finished.",
                "msg_green",
            )

        for failure in self.runtime_bus.drain_failures():
            capture_failed = failure.worker_name.startswith("capture-")
            preview_failed = failure.worker_name == "preview"
            diagnostic_failed = failure.worker_name in {
                "native-health",
                "native-pointer-recovery",
            }
            if capture_failed and not self.__is_current_capture_event(failure):
                continue
            if (
                preview_failed
                and failure.session_id is not None
                and failure.session_id != self.controller.capture.generation
            ):
                continue
            if (
                diagnostic_failed
                and failure.session_id is not None
                and failure.session_id != self.controller.diagnostic_session_id
            ):
                continue
            if (
                not capture_failed
                and not preview_failed
                and not diagnostic_failed
                and failure.session_id is not None
                and failure.session_id != self.controller.control_session_id
            ):
                continue
            if not preview_failed and not diagnostic_failed:
                self.__set_rl_buttons(
                    attached=not capture_failed,
                    running=False,
                )
            elif failure.worker_name == "native-pointer-recovery":
                self.__set_rl_buttons(attached=True, running=False)
            self.runtime_bus.log(
                f"{failure.worker_name} failed in "
                f"{failure.lifecycle_state} at "
                f"{failure.failed_at.isoformat()} "
                f"(cancelled={failure.cancellation_requested})\n"
                f"{failure.traceback}",
                "msg_red",
            )

        for alert in self.runtime_bus.drain_alerts():
            if (
                alert.session_id is not None
                and alert.session_id != self.controller.control_session_id
            ):
                continue
            self.__show_runtime_alert(alert)

        request = self.runtime_bus.pop_confirmation()
        if request is not None:
            result = self.__confirm_heading_reading(
                request.frame,
                request.angle_deg,
                request.confidence,
                request.context,
            )
            self.runtime_bus.resolve_confirmation(request, result)

        recovery = self.runtime_bus.pop_mapper_recovery()
        if recovery is not None:
            result = self.__confirm_mapper_recovery(recovery)
            self.runtime_bus.resolve_mapper_recovery(recovery, result)

    def __confirm_mapper_recovery(self, request):
        instructions = [
            sg.Text(
                f"Map: {request.map_name}",
                font="Any 12 bold",
            ),
            sg.Text(
                str(request.reason),
                size=(88, 5),
                text_color="yellow",
            ),
            sg.HorizontalSeparator(),
            sg.Text(
                "The mapper released all movement keys and saved the persistent "
                "map. Do not choose Retry In Place if you moved the character. "
                "For a full reset, manually leave/re-enter or teleport to the "
                "known spawn, keep the camera fixed, then choose Returned to Spawn.",
                size=(88, 6),
            ),
        ]
        buttons = []
        if request.can_retry_in_place:
            buttons.append(sg.Button("Retry In Place", key="retry"))
        buttons.append(sg.Button("Returned to Spawn", key="spawn"))
        buttons.append(sg.Button("Stop Mapping", key="stop"))

        window = sg.Window(
            "Mapper Recovery",
            [[item] for item in instructions] + [buttons],
            modal=True,
            keep_on_top=True,
            finalize=True,
            location=(80, 80),
        )
        try:
            while True:
                event, _values = window.read()
                if event in (sg.WIN_CLOSED, "stop"):
                    return "stop"
                if event == "retry" and request.can_retry_in_place:
                    return "retry"
                if event == "spawn":
                    return "spawn"
        finally:
            window.close()

    def __shutdown(self, bot) -> bool:
        del bot
        self.__set_status("Stopping", "Stopping workers safely…")
        results = self.controller.shutdown(timeout=8.0)
        timed_out = [kind.value for kind, stopped in results.items() if not stopped]
        if not timed_out:
            return True

        message = (
            "Shutdown timed out while waiting for "
            f"{', '.join(timed_out)}. Resources remain open; close again after "
            "the worker reports completion."
        )
        self.__set_status("Shutdown blocked", message)
        self.runtime_bus.log(message, "msg_red")
        self.__set_rl_buttons(attached=False, running=False)
        self.window["-ATTACH_WINDOW-"].update(disabled=True)
        return False

    def __is_current_status(self, key: str, status: RuntimeStatus) -> bool:
        if status.session_id is None:
            return True
        if key in {"capture_status", "preview_status"}:
            return status.session_id == self.controller.capture.generation
        if key in {"rl_status", "mapper_status"}:
            return status.session_id == self.controller.control_session_id
        return True

    def __is_current_capture_event(self, event) -> bool:
        session_id = getattr(event, "session_id", None)
        generation = self.controller.capture.generation
        if session_id is not None:
            return session_id == generation
        return not self.controller.capture_active

    def __set_rl_buttons(self, attached, running):
        """Keep the RL action buttons in a consistent state."""
        if (
            self.controller.shutdown_requested
            and not self.controller.shutdown_finalized
        ):
            # A timed-out shutdown leaves the worker manager permanently
            # closed to new work. Late completion/failure events must not
            # re-enable controls while the live worker is winding down.
            attached = False
            running = True
        self.window["-DRY_RUN-"].update(disabled=(not attached or running))
        self.window["-VALIDATE_DATA-"].update(disabled=(not attached or running))
        self.window["-START_BOT-"].update(disabled=(not attached or running))
        self.window["-RUN_AGENT-"].update(disabled=(not attached or running))
        self.window["-START_MAPPER-"].update(disabled=(not attached or running))
        self.window["-START_MANUAL_MAPPER-"].update(
            disabled=(not attached or running)
        )
        self.window["-SET_MINIMAP_ANCHOR-"].update(disabled=(not attached or running))
        self.window["-STOP_BOT-"].update(disabled=(not attached or not running))
        self.window["-ATTACH_WINDOW-"].update(disabled=running)
        self.window["-MAP-NAME-"].update(disabled=running)
        self.window["-ADD_MAP-"].update(disabled=running)
        self.window["-EDIT_MAP_MOBS-"].update(disabled=running)
        self.window["-EDIT_MAP_CELLS-"].update(disabled=running)
        self.window["-RESET_MAP-"].update(disabled=running)
        self.window["-DELETE_MAP-"].update(disabled=running)
        self.window["-NATIVE_HEALTH-"].update(disabled=not attached)
        self.window["-RECOVER_POINTERS-"].update(
            disabled=(not attached or running)
        )

    def __set_minimap_anchor(self, bot):
        if self.controller.control_active:
            sg.cprint(
                "Stop the active control task before setting the minimap center.",
                c=("white", "red"),
            )
            return

        if not bot.is_ready:
            sg.cprint(
                "Attach the Flyff window first.",
                c=("white", "red"),
            )
            return

        try:
            from mapper.MinimapAnchorSetup import MinimapAnchorSetup

            sg.cprint(
                "Opening minimap-center selector. Click the exact center of "
                "the player arrow, then press Enter.",
                c=("white", "blue"),
            )
            setup = MinimapAnchorSetup(bot)
            output_path = setup.run()
            sg.cprint(
                f"Fixed minimap center saved to {output_path}.",
                c=("white", "green"),
            )
        except Exception as error:  # noqa: BLE001 - GUI command boundary.
            sg.cprint(
                f"Minimap-center setup failed: {error}",
                c=("white", "red"),
            )

    def __clear_manual_calibration_artifacts(self):
        """
        Manual recalibration starts a new diagnostic session.

        Preserve minimap_anchor.json and the last valid calibration. The
        calibrator replaces calibration.json only after a complete successful
        run, so a cancelled or failed attempt must not remove the working
        configuration.
        """
        import shutil
        from pathlib import Path

        project_root = Path(__file__).resolve().parent
        debug_directories = [
            project_root / "debug" / "minimap_heading",
        ]

        removed = []
        for directory in debug_directories:
            if directory.exists():
                shutil.rmtree(directory)
                removed.append(str(directory))
            directory.mkdir(parents=True, exist_ok=True)

        if removed:
            sg.cprint(
                "Manual recalibration cleared old minimap-heading debug files; "
                "the previous valid calibration is kept until this run succeeds.",
                c=("white", "blue"),
            )
        else:
            sg.cprint(
                "Manual recalibration started; the previous valid calibration "
                "is kept until this run succeeds.",
                c=("white", "blue"),
            )

    def close(self):
        if self.controller is None:
            self.runtime_bus.close()
        self.window.close()

    def __show_runtime_alert(self, alert: RuntimeAlert) -> None:
        if winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except RuntimeError:
                pass
        sg.popup_error(
            alert.message,
            title=alert.title,
            keep_on_top=True,
            modal=True,
            location=(80, 80),
        )

    def show_error(self, message: str) -> None:
        """Display an error above the main window from the GUI thread."""
        sg.popup_error(
            str(message),
            title="Flyff FVF Error",
            keep_on_top=True,
            modal=True,
            location=(80, 80),
        )

    def __load_settings(self, bot):
        show_frames = sg.user_settings_get_entry("-SHOW_FRAMES-", True)
        self.window["-SHOW_FRAMES-"].update(show_frames)
        self.window["-SHOW_MATCHES_TEXT-"].update(visible=show_frames)
        self.window["-SHOW_BOXES-"].update(visible=show_frames)
        self.window["-SHOW_MARKERS-"].update(visible=show_frames)
        self.window["-VISION_FRAME-"].update(visible=show_frames)
        bot.set_config(show_frames=show_frames)

        show_matches_text = sg.user_settings_get_entry("-SHOW_MATCHES_TEXT-", False)
        self.window["-SHOW_MATCHES_TEXT-"].update(show_matches_text)
        bot.set_config(show_matches_text=show_matches_text)

        show_mobs_pos_boxes = sg.user_settings_get_entry("-SHOW_BOXES-", False)
        self.window["-SHOW_BOXES-"].update(show_mobs_pos_boxes)
        bot.set_config(show_mobs_pos_boxes=show_mobs_pos_boxes)

        show_mobs_pos_markers = sg.user_settings_get_entry("-SHOW_MARKERS-", True)
        self.window["-SHOW_MARKERS-"].update(show_mobs_pos_markers)
        bot.set_config(show_mobs_pos_markers=show_mobs_pos_markers)

        bot.set_config(
            mob_pos_match_threshold=0.7,
            mob_still_alive_match_threshold=0.7,
            mob_existence_match_threshold=0.7,
            inventory_perin_converter_match_threshold=0.7,
            inventory_icons_match_threshold=0.7,
            mobs_kill_goal=None,
            fight_time_limit_sec=8,
            delay_to_check_mob_still_alive_sec=0.25,
        )

        convert_timer = sg.user_settings_get_entry(
            "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-", "30"
        )
        self.window["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"].update(convert_timer)
        bot.set_config(convert_penya_to_perins_timer_min=convert_timer)

        saved_map = sg.user_settings_get_entry(
            "saved_map_name",
            self.map_catalog.default_name,
        )
        if saved_map not in self.map_names:
            saved_map = self.map_catalog.default_name
        self.__apply_map_selection(bot, saved_map, publish_preview=True)

    def __apply_map_selection(self, bot, map_name, *, publish_preview):
        try:
            profile = self.map_catalog.get(map_name)
        except (ValueError, TypeError) as error:
            self.runtime_bus.log(str(error), "msg_red")
            return

        self._selected_map_name = profile.name
        bot.set_config(selected_map_name=profile.name)
        sg.user_settings_set_entry("saved_map_name", profile.name)
        if hasattr(self, "window"):
            self.window["-MAP-NAME-"].update(profile.name)

        all_mobs = bot.get_all_mobs()
        allowed_names = [name for name in profile.mobs if name in all_mobs]
        settings_key = f"saved_selected_mobs::{profile.slug}"
        selected_names = sg.user_settings_get_entry(settings_key, None)
        if selected_names is None:
            selected_names = list(allowed_names)
        selected_names = [name for name in selected_names if name in allowed_names]
        bot.set_config(selected_mobs=[all_mobs[name] for name in selected_names])
        sg.user_settings_set_entry(settings_key, selected_names)
        sg.user_settings_set_entry("saved_selected_mobs", selected_names)

        mob_text = ", ".join(allowed_names) if allowed_names else "No registered mobs"
        if hasattr(self, "window"):
            self.window["-MAP-MOBS-"].update(f"Mobs: {mob_text}")

        if publish_preview and self.controller is not None:
            if not self.controller.publish_map_preview(profile.name):
                self.__clear_live_map(
                    f"{profile.name}: no saved map yet — mapping will create it."
                )

    def __reload_map_catalog(self, *, selected_name=None):
        self.map_catalog = MapCatalog()
        self.map_names = list(self.map_catalog.names())
        selected = selected_name
        if selected not in self.map_names:
            selected = self.map_catalog.default_name
        self._selected_map_name = selected
        if hasattr(self, "window"):
            self.window["-MAP-NAME-"].update(
                values=self.map_names,
                value=selected,
            )
        return selected

    def __edit_map_cells_popup(self):
        if not self.__map_management_available():
            return
        if self.controller is None:
            return

        profile = self.map_catalog.get(self._selected_map_name)
        directory = self.map_catalog.map_directory(profile.name)
        if not (directory / "map.json").is_file():
            self.show_error(
                f"{profile.name} does not have a saved map yet. Run the mapper first."
            )
            return
        grid, warning = OccupancyGrid.load(directory)
        if warning is not None:
            self.show_error(warning)
            return

        session = ManualMapEditorSession(grid)
        radius = 35

        def cell_pixels_for(view_radius):
            return max(4, min(11, 620 // (2 * int(view_radius) + 1)))

        cell_pixels = cell_pixels_for(radius)

        def encoded_view():
            image = session.render_view(
                radius_cells=radius,
                cell_pixels=cell_pixels,
            )
            encoded = cv.imencode(".png", image)
            return b"" if not encoded[0] else encoded[1].tobytes()

        layout = [
            [
                sg.Text(f"Occupancy cell editor — {profile.name}", font="Any 13 bold"),
                sg.Push(),
                sg.Text("View radius:"),
                sg.Combo(
                    [20, 30, 35, 40, 50, 60],
                    default_value=radius,
                    readonly=True,
                    enable_events=True,
                    key="-CELL-EDIT-RADIUS-",
                    size=(5, 1),
                ),
            ],
            [
                sg.Frame(
                    "Paint",
                    [[
                        sg.Radio("Explored / free", "CELL_TOOL", default=True, key="-CELL-FREE-"),
                        sg.Radio("Blocked", "CELL_TOOL", key="-CELL-BLOCKED-"),
                        sg.Radio(
                            "Teleport area",
                            "CELL_TOOL",
                            key="-CELL-TELEPORT-",
                            text_color="red",
                        ),
                        sg.Radio("Clear to unknown", "CELL_TOOL", key="-CELL-ERASE-"),
                    ]],
                ),
                sg.Frame(
                    "Selection",
                    [[
                        sg.Radio("Line / brush", "CELL_SHAPE", default=True, key="-CELL-LINE-"),
                        sg.Radio("Rectangle", "CELL_SHAPE", key="-CELL-RECT-"),
                    ]],
                ),
            ],
            [
                sg.Button("↖", key="-CELL-PAN-NW-"),
                sg.Button("↑", key="-CELL-PAN-N-"),
                sg.Button("↗", key="-CELL-PAN-NE-"),
                sg.Button("←", key="-CELL-PAN-W-"),
                sg.Button("Player", key="-CELL-CENTER-"),
                sg.Button("→", key="-CELL-PAN-E-"),
                sg.Button("↙", key="-CELL-PAN-SW-"),
                sg.Button("↓", key="-CELL-PAN-S-"),
                sg.Button("↘", key="-CELL-PAN-SE-"),
                sg.Push(),
                sg.Button("Undo", key="-CELL-UNDO-"),
            ],
            [
                sg.Image(
                    data=encoded_view(),
                    key="-CELL-EDIT-IMAGE-",
                    background_color="#5a5a5a",
                )
            ],
            [
                sg.Text(
                    "Drag to select. Saved edits become authoritative free, blocked, red "
                    "teleport, or unknown map cells. Manual-drive mode stops before entering "
                    "a red teleport cell.",
                    size=(90, 2),
                    key="-CELL-EDIT-STATUS-",
                )
            ],
            [
                sg.Button("Save Cells", key="-CELL-SAVE-", button_color=("white", "#287a3c")),
                sg.Button("Cancel"),
            ],
        ]
        editor = sg.Window(
            "Edit Occupancy Cells",
            layout,
            modal=True,
            keep_on_top=False,
            finalize=True,
            resizable=False,
        )
        image_widget = editor["-CELL-EDIT-IMAGE-"].Widget
        image_widget.bind(
            "<ButtonPress-1>",
            lambda event: editor.write_event_value("-CELL-MOUSE-DOWN-", (event.x, event.y)),
        )
        image_widget.bind(
            "<B1-Motion>",
            lambda event: editor.write_event_value("-CELL-MOUSE-DRAG-", (event.x, event.y)),
        )
        image_widget.bind(
            "<ButtonRelease-1>",
            lambda event: editor.write_event_value("-CELL-MOUSE-UP-", (event.x, event.y)),
        )

        drag_start = None
        drag_end = None

        def selected_mode(values):
            if values.get("-CELL-BLOCKED-"):
                return "blocked"
            if values.get("-CELL-TELEPORT-"):
                return "teleport"
            if values.get("-CELL-ERASE-"):
                return "erase"
            return "free"

        def refresh(message=None):
            editor["-CELL-EDIT-IMAGE-"].update(data=encoded_view())
            if message is not None:
                editor["-CELL-EDIT-STATUS-"].update(message)

        pan_directions = {
            "-CELL-PAN-NW-": (-1, 1),
            "-CELL-PAN-N-": (0, 1),
            "-CELL-PAN-NE-": (1, 1),
            "-CELL-PAN-W-": (-1, 0),
            "-CELL-PAN-E-": (1, 0),
            "-CELL-PAN-SW-": (-1, -1),
            "-CELL-PAN-S-": (0, -1),
            "-CELL-PAN-SE-": (1, -1),
        }

        try:
            while True:
                event, values = editor.read()
                if event in (sg.WIN_CLOSED, "Cancel"):
                    return
                if event == "-CELL-EDIT-RADIUS-":
                    radius = int(values[event])
                    cell_pixels = cell_pixels_for(radius)
                    refresh()
                    continue
                if event in pan_directions:
                    dx, dy = pan_directions[event]
                    step = max(5, radius // 2)
                    session.pan(dx * step, dy * step)
                    refresh()
                    continue
                if event == "-CELL-CENTER-":
                    session.center_on_player()
                    refresh()
                    continue
                if event == "-CELL-UNDO-":
                    if session.undo():
                        refresh(f"Undid the last selection. {len(session.staged)} staged cells remain.")
                    continue
                if event == "-CELL-MOUSE-DOWN-":
                    drag_start = session.pixel_to_cell(
                        *values[event], radius_cells=radius, cell_pixels=cell_pixels
                    )
                    drag_end = drag_start
                    continue
                if event == "-CELL-MOUSE-DRAG-":
                    drag_end = session.pixel_to_cell(
                        *values[event], radius_cells=radius, cell_pixels=cell_pixels
                    )
                    if drag_end is not None:
                        editor["-CELL-EDIT-STATUS-"].update(
                            f"Selection endpoint: {drag_end}. Release to stage cells."
                        )
                    continue
                if event == "-CELL-MOUSE-UP-":
                    drag_end = session.pixel_to_cell(
                        *values[event], radius_cells=radius, cell_pixels=cell_pixels
                    )
                    if drag_start is None or drag_end is None:
                        drag_start = drag_end = None
                        continue
                    cells = (
                        session.rectangle_cells(drag_start, drag_end)
                        if values.get("-CELL-RECT-")
                        else session.line_cells(drag_start, drag_end)
                    )
                    try:
                        changed = session.stage_cells(cells, selected_mode(values))
                    except ValueError as error:
                        self.show_error(str(error))
                    else:
                        refresh(
                            f"Staged {changed} cells; {len(session.staged)} total. "
                            "Press Save Cells to persist."
                        )
                    drag_start = drag_end = None
                    continue
                if event == "-CELL-SAVE-":
                    if not session.staged:
                        editor["-CELL-EDIT-STATUS-"].update("No cells are staged.")
                        continue
                    try:
                        summary = self.controller.apply_manual_map_edits(
                            profile.name,
                            session.staged,
                        )
                    except Exception as error:  # noqa: BLE001 - GUI command boundary.
                        self.show_error(f"Could not save manual map cells: {error}")
                        continue
                    message = (
                        f"Saved {summary.free_cells} free, {summary.blocked_cells} blocked, "
                        f"{summary.teleport_cells} teleport, and cleared "
                        f"{summary.erased_cells} cells to unknown"
                    )
                    if summary.skipped_cells:
                        message += f"; skipped {summary.skipped_cells} unchanged/invalid cells"
                    message += "."
                    self.runtime_bus.log(message, "msg_green")
                    return
        finally:
            editor.close()

    def __map_management_available(self):
        if self.controller is not None and self.controller.control_active:
            self.runtime_bus.log(
                "Stop the active control task before changing map profiles.",
                "msg_red",
            )
            return False
        return True

    def __add_map_popup(self, bot):
        if not self.__map_management_available():
            return

        registered_mobs = sorted(bot.get_all_mobs())
        popup = sg.Window(
            "Add Map",
            [
                [sg.Text("Map name:"), sg.Input(key="-NEW-MAP-NAME-", expand_x=True)],
                [sg.Text("Mobs available on this map:")],
                [
                    sg.Listbox(
                        values=registered_mobs,
                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                        size=(48, min(12, max(4, len(registered_mobs)))),
                        key="-NEW-MAP-MOBS-",
                    )
                ],
                [sg.Button("Create"), sg.Button("Cancel")],
            ],
            modal=True,
            finalize=True,
        )
        try:
            while True:
                event, values = popup.read()
                if event in (sg.WIN_CLOSED, "Cancel"):
                    return
                if event != "Create":
                    continue
                try:
                    profile = self.map_catalog.create_map(
                        values.get("-NEW-MAP-NAME-", ""),
                        mobs=values.get("-NEW-MAP-MOBS-", []),
                    )
                except (OSError, ValueError) as error:
                    self.show_error(str(error))
                    continue
                self.__reload_map_catalog(selected_name=profile.name)
                self.__apply_map_selection(
                    bot,
                    profile.name,
                    publish_preview=True,
                )
                self.runtime_bus.log(
                    f"Created map profile '{profile.name}'.",
                    "msg_green",
                )
                return
        finally:
            popup.close()

    def __edit_map_mobs_popup(self, bot):
        if not self.__map_management_available():
            return
        profile = self.map_catalog.get(self._selected_map_name)
        registered_mobs = sorted(bot.get_all_mobs())
        popup = sg.Window(
            "Edit Map Mobs",
            [
                [sg.Text(f"Mobs available on '{profile.name}':")],
                [
                    sg.Listbox(
                        values=registered_mobs,
                        default_values=[
                            name for name in profile.mobs if name in registered_mobs
                        ],
                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                        size=(48, min(12, max(4, len(registered_mobs)))),
                        key="-EDIT-MAP-MOBS-",
                    )
                ],
                [sg.Button("Save"), sg.Button("Cancel")],
            ],
            modal=True,
            finalize=True,
        )
        try:
            event, values = popup.read()
            if event != "Save":
                return
            try:
                self.map_catalog.update_mobs(
                    profile.name,
                    values.get("-EDIT-MAP-MOBS-", []),
                )
            except (OSError, ValueError) as error:
                self.show_error(f"Could not update map mobs: {error}")
                return
        finally:
            popup.close()

        self.__reload_map_catalog(selected_name=profile.name)
        self.__apply_map_selection(bot, profile.name, publish_preview=False)
        self.runtime_bus.log(
            f"Updated available mobs for '{profile.name}'.",
            "msg_green",
        )

    def __reset_selected_map(self, bot):
        if not self.__map_management_available():
            return
        name = self._selected_map_name
        answer = sg.popup_yes_no(
            f"Reset all persistent mapping progress for '{name}'?\n\n"
            "The named map and its mob list will remain. Diagnostic run folders "
            "will be kept. This cannot be undone.",
            title="Reset Map Progress",
            keep_on_top=True,
        )
        if answer != "Yes":
            return
        try:
            self.map_catalog.reset_map(name)
        except (OSError, ValueError) as error:
            self.show_error(f"Could not reset map: {error}")
            return
        self.__clear_live_map(f"{name}: map progress reset — next run starts empty.")
        self.__apply_map_selection(bot, name, publish_preview=False)
        self.runtime_bus.log(
            f"Reset persistent mapping progress for '{name}'.",
            "msg_green",
        )

    def __delete_selected_map(self, bot):
        if not self.__map_management_available():
            return
        name = self._selected_map_name
        if len(self.map_names) <= 1:
            self.show_error("At least one map profile must remain.")
            return

        popup = sg.Window(
            "Delete Map",
            [
                [sg.Text(f"Delete map profile '{name}' and its persistent map?")],
                [
                    sg.Checkbox(
                        "Also delete diagnostic mapping-run history",
                        default=False,
                        key="-DELETE-MAP-RUNS-",
                    )
                ],
                [
                    sg.Button(
                        "Delete",
                        button_color=("white", "#a83232"),
                    ),
                    sg.Button("Cancel"),
                ],
            ],
            modal=True,
            finalize=True,
        )
        try:
            event, values = popup.read()
            if event != "Delete":
                return
            try:
                selected = self.map_catalog.delete_map(
                    name,
                    delete_run_history=bool(values.get("-DELETE-MAP-RUNS-", False)),
                )
            except (OSError, ValueError) as error:
                self.show_error(f"Could not delete map: {error}")
                return
        finally:
            popup.close()

        selected = self.__reload_map_catalog(selected_name=selected)
        self.__apply_map_selection(bot, selected, publish_preview=True)
        self.runtime_bus.log(f"Deleted map profile '{name}'.", "msg_green")

    def __set_hotkeys(self):
        self.window.bind("<Alt_L><s>", "-STOP_BOT-")

    def __get_layout(self):
        title = [sg.Text("Flyff FVF", font="Any 18")]

        actions = sg.Frame(
            "Actions:",
            [
                [
                    sg.Button(
                        "Attach Window",
                        key="-ATTACH_WINDOW-",
                        expand_x=True,
                    )
                ],
                [
                    sg.Button(
                        "Native Dry Run (No Learning)",
                        disabled=True,
                        key="-DRY_RUN-",
                        expand_x=True,
                    )
                ],
                [
                    sg.Button(
                        "Validate Training Data (No Learning)",
                        disabled=True,
                        key="-VALIDATE_DATA-",
                        expand_x=True,
                        tooltip=(
                            "Runs the live training input pipeline and creates a "
                            "SEND_TO_CHATGPT debug ZIP."
                        ),
                    )
                ],
                [
                    sg.Button(
                        "Start Training",
                        disabled=True,
                        key="-START_BOT-",
                        expand_x=True,
                    ),
                    sg.Button(
                        "Run Trained Agent",
                        disabled=True,
                        key="-RUN_AGENT-",
                        expand_x=True,
                    ),
                ],
                [
                    sg.Button(
                        "Calibrate Minimap (optional)",
                        disabled=True,
                        key="-SET_MINIMAP_ANCHOR-",
                        expand_x=True,
                    ),
                ],
                [
                    sg.Text(
                        "Player HP is read automatically from the status panel.",
                        tooltip="Keep the full player-status panel visible.",
                    ),
                ],
                [
                    sg.Button(
                        "Native Health",
                        disabled=True,
                        key="-NATIVE_HEALTH-",
                        expand_x=True,
                    ),
                    sg.Button(
                        "Recover Pointers",
                        disabled=True,
                        key="-RECOVER_POINTERS-",
                        expand_x=True,
                    ),
                ],
                [
                    sg.Button(
                        "Map Area (Automatic)",
                        disabled=True,
                        key="-START_MAPPER-",
                        expand_x=True,
                    )
                ],
                [
                    sg.Button(
                        "Trace Map While I Drive",
                        disabled=True,
                        key="-START_MANUAL_MAPPER-",
                        expand_x=True,
                    )
                ],
                [
                    sg.Button(
                        "Stop (Alt+s)",
                        disabled=True,
                        key="-STOP_BOT-",
                        expand_x=True,
                    ),
                    sg.Button("Exit", expand_x=True),
                ],
                [sg.Text("Attached window:", font="Any 8")],
                [
                    sg.Text(
                        "",
                        font="Any 8",
                        text_color="red",
                        key="-ATTACHED_WINDOW-",
                        size=(34, 2),
                    )
                ],
            ],
            expand_x=True,
        )

        map_config = sg.Frame(
            "Map:",
            [
                [
                    sg.Text("Selected map:"),
                    sg.Combo(
                        self.map_names,
                        default_value=self._selected_map_name,
                        readonly=True,
                        enable_events=True,
                        key="-MAP-NAME-",
                        expand_x=True,
                    ),
                ],
                [
                    sg.Text(
                        "Mobs: --",
                        key="-MAP-MOBS-",
                        size=(38, 2),
                    )
                ],
                [
                    sg.Button("Add Map", key="-ADD_MAP-"),
                    sg.Button("Edit Map Mobs", key="-EDIT_MAP_MOBS-"),
                ],
                [
                    sg.Button("Edit Map Cells", key="-EDIT_MAP_CELLS-", expand_x=True),
                ],
                [
                    sg.Button("Reset Progress", key="-RESET_MAP-"),
                    sg.Button(
                        "Delete Map",
                        key="-DELETE_MAP-",
                        button_color=("white", "#a83232"),
                    ),
                ],
            ],
            expand_x=True,
        )

        mobs_config = sg.Frame(
            "Mobs Configuration:",
            [
                [
                    sg.Button("Select Mobs", key="-SELECT_MOBS-"),
                    sg.Button("Add Mob", key="-ADD_MOB-"),
                    sg.Button("Delete Mob", key="-DELETE_MOB-"),
                ]
            ],
            expand_x=True,
        )

        options = sg.Frame(
            "Options:",
            [
                [
                    sg.Checkbox(
                        "Show bot's vision",
                        True,
                        enable_events=True,
                        key="-SHOW_FRAMES-",
                    )
                ],
                [
                    sg.pin(
                        sg.Checkbox(
                            "Show matches text",
                            False,
                            enable_events=True,
                            key="-SHOW_MATCHES_TEXT-",
                        )
                    )
                ],
                [
                    sg.pin(
                        sg.Checkbox(
                            "Show mobs boxes",
                            False,
                            enable_events=True,
                            key="-SHOW_BOXES-",
                        )
                    )
                ],
                [
                    sg.pin(
                        sg.Checkbox(
                            "Show mobs markers",
                            True,
                            enable_events=True,
                            key="-SHOW_MARKERS-",
                        )
                    )
                ],
                [sg.HorizontalSeparator()],
                [sg.Text("Timer to convert penya to perins (m):")],
                [
                    sg.InputText(
                        "30",
                        size=(10, 1),
                        enable_events=True,
                        key="-CONVERT_PENYA_TO_PERINS_TIMER_MIN-",
                    )
                ],
            ],
            expand_x=True,
        )

        status = sg.Frame(
            "Status:",
            [
                [
                    sg.Text(
                        "Mode: Idle",
                        key="-STATUS_MODE-",
                        size=(18, 1),
                    ),
                    sg.Text("FPS: --", key="-VIDEO_FPS-", size=(10, 1)),
                ],
                [
                    sg.Text(
                        "Ready",
                        key="-STATUS_MESSAGE-",
                        size=(38, 3),
                    )
                ],
                [
                    sg.Button(
                        "Show Log",
                        key="-SHOW_LOG-",
                        expand_x=True,
                    )
                ],
                [
                    sg.Multiline(
                        size=(1, 1),
                        key="-ML-",
                        autoscroll=True,
                        visible=False,
                    )
                ],
            ],
            expand_x=True,
        )

        controls = sg.Column(
            [
                [actions],
                [map_config],
                [mobs_config],
                [options],
                [status],
            ],
            size=(335, 820),
            pad=((0, 8), (0, 0)),
            scrollable=True,
            vertical_scroll_only=True,
            expand_y=True,
            key="-MAIN_COLUMN-",
        )

        bot_vision = sg.Frame(
            "Bot Vision:",
            [
                [
                    sg.Text("Image Resolution:"),
                    sg.Combo(
                        list(self.frame_resolutions.keys()),
                        default_value="Fit panel",
                        readonly=True,
                        key="-DEBUG_IMG_WIDTH-",
                    ),
                ],
                [
                    sg.Image(
                        filename="",
                        key="-DEBUG_IMAGE-",
                        size=(1, 1),
                        expand_x=True,
                        expand_y=True,
                    )
                ],
            ],
            visible=True,
            expand_x=True,
            expand_y=True,
            key="-VISION_FRAME-",
        )

        live_map = sg.Frame(
            "Live Map:",
            [
                [
                    sg.Text("Unknown", text_color="gray"),
                    sg.Text("Explored", text_color="white"),
                    sg.Text("Wall / obstacle", text_color="black"),
                    sg.Text("Teleport", text_color="red"),
                    sg.Text("Player", text_color="yellow"),
                    sg.Text("Detected monster (square)", text_color="light green"),
                ],
                [
                    sg.Image(
                        data=self.__make_live_map_placeholder(
                            "Select a map to load its persistent progress."
                        ),
                        key="-MAPPER_IMAGE-",
                        size=(960, 245),
                    )
                ],
            ],
            expand_x=True,
            key="-MAPPER_FRAME-CONTAINER-",
        )

        visuals = sg.Column(
            [
                [bot_vision],
                [live_map],
            ],
            pad=(0, 0),
            expand_x=True,
            expand_y=True,
        )

        return [title, [controls, visuals]]

    def __preview_target_size(self, selected_resolution):
        configured = self.frame_resolutions.get(selected_resolution)
        if configured is not None:
            return configured

        # PySimpleGUI's Image element is backed by a Tk Label. With expansion
        # enabled its actual widget size follows the current bot-view panel,
        # so fullscreen and small-window previews use the same available area
        # rather than the game capture's source resolution.
        try:
            widget = self.window["-DEBUG_IMAGE-"].Widget
            width = int(widget.winfo_width())
            height = int(widget.winfo_height())
        except Exception:
            width, height = 0, 0

        if width < 64 or height < 64:
            try:
                frame_widget = self.window["-VISION_FRAME-"].Widget
                width = max(64, int(frame_widget.winfo_width()) - 18)
                height = max(64, int(frame_widget.winfo_height()) - 62)
            except Exception:
                width, height = 960, 540

        # Avoid transient one-pixel or absurd values while Tk is relaying out
        # a resized window. The next preview frame will use the final size.
        width = max(160, min(width, 3840))
        height = max(90, min(height, 2160))
        return width, height

    def __fit_image(self, image, target_width, target_height):
        """Letterbox an image without changing its aspect ratio."""
        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]
        scale = min(
            target_width / max(1, width),
            target_height / max(1, height),
        )
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv.resize(
            image,
            (resized_width, resized_height),
            interpolation=(cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR),
        )

        top = (target_height - resized_height) // 2
        bottom = target_height - resized_height - top
        left = (target_width - resized_width) // 2
        right = target_width - resized_width - left

        return cv.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv.BORDER_CONSTANT,
            value=(20, 20, 20),
        )

    def __set_status(self, mode, message):
        self._status_mode = mode
        self._last_status_message = str(message)
        if hasattr(self, "window"):
            self.window["-STATUS_MODE-"].update(f"Mode: {mode}")
            self.window["-STATUS_MESSAGE-"].update(self._last_status_message[-240:])

    def __show_log_window(self):
        if self._log_window is not None:
            self._log_window.bring_to_front()
            return
        log_text = self.window["-ML-"].get()

        self._log_window = sg.Window(
            "Flyff FVF Log",
            [
                [
                    sg.Multiline(
                        log_text,
                        key="-LOG-TEXT-",
                        size=(110, 35),
                        disabled=True,
                        autoscroll=True,
                    )
                ],
                [sg.Button("Close")],
            ],
            resizable=True,
            finalize=True,
        )

    def __service_log_window(self):
        if self._log_window is None:
            return
        event, _ = self._log_window.read(timeout=0)
        if event in (sg.WIN_CLOSED, "Close"):
            self._log_window.close()
            self._log_window = None

    def __make_live_map_placeholder(self, message):
        import numpy as np

        canvas = np.full((245, 960, 3), 24, dtype=np.uint8)
        cv.rectangle(canvas, (1, 1), (958, 243), (70, 70, 70), 1)

        title = "LIVE MAP"
        cv.putText(
            canvas,
            title,
            (40, 92),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 180, 180),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            canvas,
            message,
            (40, 145),
            cv.FONT_HERSHEY_SIMPLEX,
            0.62,
            (150, 150, 150),
            1,
            cv.LINE_AA,
        )
        return cv.imencode(".png", canvas)[1].tobytes()

    def __clear_live_map(self, message):
        if hasattr(self, "window"):
            self.window["-MAPPER_IMAGE-"].update(
                data=self.__make_live_map_placeholder(message)
            )

    def __confirm_heading_reading(
        self,
        frame,
        angle_deg,
        confidence,
        context,
    ):
        """
        Ask the user whether one detected heading is visually correct.

        Returns:
            True  -> accept
            False -> reject and reacquire
            None  -> stop calibration
        """
        preview = frame.copy()
        if preview.ndim == 2 or (preview.ndim == 3 and preview.shape[2] == 1):
            preview = cv.cvtColor(preview, cv.COLOR_GRAY2BGR)
        try:
            if self._heading_overlay_detector is None:
                from mapper import MinimapHeadingDetector

                self._heading_overlay_detector = MinimapHeadingDetector()

            anchor = self._heading_overlay_detector._load_anchor(preview)
            center = (
                int(anchor["arrow_center_x"]),
                int(anchor["arrow_center_y"]),
            )
            radians = math.radians(float(angle_deg))
            endpoint = (
                round(center[0] + math.sin(radians) * 70),
                round(center[1] - math.cos(radians) * 70),
            )
            cv.circle(preview, center, 8, (0, 255, 255), 2)
            cv.arrowedLine(
                preview,
                center,
                endpoint,
                (0, 255, 255),
                4,
                tipLength=0.25,
            )
            cv.putText(
                preview,
                f"{float(angle_deg):.1f} deg  conf {float(confidence):.2f}",
                (20, 34),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )
        except Exception as error:  # noqa: BLE001 - diagnostic overlay boundary.
            self.runtime_bus.log(
                f"Heading confirmation overlay failed: {error}",
                "msg_red",
            )

        preview = self.__fit_image(preview, 960, 540)
        png = cv.imencode(".png", preview)[1].tobytes()

        window = sg.Window(
            "Verify Detected Heading",
            [
                [
                    sg.Text(
                        str(context),
                        size=(100, 2),
                    )
                ],
                [sg.Image(data=png)],
                [
                    sg.Text(
                        "Does the yellow arrow match the character's actual heading?"
                    )
                ],
                [
                    sg.Button("Correct", key="-HEADING_CORRECT-"),
                    sg.Button("Incorrect — Re-read", key="-HEADING_REJECT-"),
                    sg.Button("Stop Calibration", key="-HEADING_STOP-"),
                ],
            ],
            modal=True,
            finalize=True,
            keep_on_top=True,
        )

        result = None
        while True:
            event, _ = window.read()
            if event == "-HEADING_CORRECT-":
                result = True
                break
            if event == "-HEADING_REJECT-":
                result = False
                break
            if event in (sg.WIN_CLOSED, "-HEADING_STOP-"):
                result = None
                break

        window.close()
        return result

    def __attach_window_popup(self):
        handlers = self.__flyff_window_handlers()
        titles = list(handlers)
        popup_window = sg.Window(
            "Attach Window",
            [
                [sg.Text("Please select the window to attach to:")],
                [
                    sg.DropDown(
                        titles,
                        default_value=titles[0] if titles else "",
                        readonly=True,
                        key="-DROP-",
                    ),
                    sg.Button("Refresh"),
                ],
                [sg.OK(), sg.Cancel()],
            ],
            size=(400, 100),
        )
        while True:
            event, values = popup_window.read()
            if event == "Refresh":
                handlers = self.__flyff_window_handlers()
                titles = list(handlers)
                popup_window["-DROP-"].update(
                    values=titles,
                    value=titles[0] if titles else "",
                )
            if event in (sg.WIN_CLOSED, "Cancel"):
                popup_window.close()
                return None, None
            if event == "OK":
                title = values.get("-DROP-", "")
                if title not in handlers:
                    self.show_error(
                        "No Flyff window is selected. Open the game or click Refresh."
                    )
                    continue
                popup_window.close()
                return title, handlers[title]

    @staticmethod
    def __flyff_window_handlers() -> dict[str, int]:
        return {
            title: handle
            for title, handle in get_window_handlers().items()
            if title.startswith("Spirit Of Madrigal")
        }

    def __select_mobs_popup(self, bot, is_delete_form=False):
        registered_mobs = bot.get_all_mobs()
        if is_delete_form:
            all_mobs = registered_mobs
        else:
            allowed = set(self.map_catalog.mobs_for(self._selected_map_name))
            all_mobs = {
                name: params
                for name, params in registered_mobs.items()
                if name in allowed
            }
        selected_mobs_names = [
            mob["name"]
            for mob in bot.config["selected_mobs"]
            if mob["name"] in all_mobs
        ]

        all_mobs_titles = [
            f"{name} - {params['element']} - {params['map_name']}"
            for (name, params) in dict.items(all_mobs)
        ]
        selected_mobs_titles = [
            f"{name} - {params['element']} - {params['map_name']}"
            for (name, params) in dict.items(all_mobs)
            if name in selected_mobs_names
        ]
        if is_delete_form:
            selected_mobs_titles = []
        last_highlighted_mob = None

        popup_window = sg.Window(
            "Select Mobs" if not is_delete_form else "Delete mobs",
            [
                [
                    sg.Text(
                        f"Select the mobs to {'kill' if not is_delete_form else 'delete'}:"
                    )
                ],
                [
                    sg.Text("Find: "),
                    sg.Input(enable_events=True, expand_x=True, key="-MOBS_SEARCH-"),
                ],
                [
                    sg.Listbox(
                        values=all_mobs_titles,
                        default_values=selected_mobs_titles,
                        size=(60, 10),
                        enable_events=True,
                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                        key="-MOBS_LIST-",
                    )
                ],
                [
                    sg.Button("Reset"),
                    sg.Button(
                        "Save" if not is_delete_form else "Delete",
                        button_color=None if not is_delete_form else "#ea4335",
                    ),
                ],
            ],
        )
        listbox = popup_window["-MOBS_LIST-"]
        while True:
            event, values = popup_window.read()

            if event == sg.WIN_CLOSED:
                popup_window.close()
                return [], []

            if values["-MOBS_SEARCH-"] != "":
                search = values["-MOBS_SEARCH-"]
                best_match = difflib.get_close_matches(
                    search, all_mobs_titles, n=1, cutoff=0.0
                )
                if last_highlighted_mob is not None:
                    listbox.Widget.itemconfigure(
                        last_highlighted_mob, bg=listbox.BackgroundColor
                    )
                    last_highlighted_mob = None
                if len(best_match) > 0:
                    best_match_index = all_mobs_titles.index(best_match[0])
                    listbox.Widget.itemconfigure(
                        best_match_index, bg=hex_variant(listbox.BackgroundColor, -20)
                    )
                    listbox.update(scroll_to_index=best_match_index)
                    last_highlighted_mob = best_match_index
            else:
                if last_highlighted_mob is not None:
                    listbox.Widget.itemconfigure(
                        last_highlighted_mob, bg=listbox.BackgroundColor
                    )
                    last_highlighted_mob = None

            if event == "-MOBS_LIST-" and len(values["-MOBS_LIST-"]):
                selected_mobs_indexes = [
                    all_mobs_titles.index(mob) for mob in values["-MOBS_LIST-"]
                ]
                listbox.update(set_to_index=selected_mobs_indexes)

            if event == "Reset":
                listbox.update(set_to_index=[])
            if event == "Save":
                selected_mobs_indexes = [
                    all_mobs_titles.index(mob) for mob in values["-MOBS_LIST-"]
                ]
                popup_window.close()
                all_names = list(dict.keys(all_mobs))
                selected_names = [all_names[i] for i in selected_mobs_indexes]
                profile = self.map_catalog.get(self._selected_map_name)
                sg.user_settings_set_entry(
                    f"saved_selected_mobs::{profile.slug}",
                    selected_names,
                )
                sg.user_settings_set_entry("saved_selected_mobs", selected_names)
                bot.set_config(
                    selected_mobs=[all_mobs[name] for name in selected_names]
                )
                return
            if event == "Delete":
                from assets.Assets import MobInfo

                deleted_mobs_names = [
                    mob.split("-")[0].strip() for mob in values["-MOBS_LIST-"]
                ]
                popup_window.close()
                # unselect deleted mobs if they were selected
                bot.set_config(
                    selected_mobs=[
                        all_mobs[name]
                        for name in selected_mobs_names
                        if name not in deleted_mobs_names
                    ]
                )
                MobInfo.delete_mobs(deleted_mobs_names)
                return

    def __add_mob_popup(self, bot):
        from assets.Assets import (
            mob_type_electricity_path,
            mob_type_fire_path,
            mob_type_soil_path,
            mob_type_water_path,
            mob_type_wind_path,
        )

        element_buttons_layout = [
            sg.Text("Select mob element: "),
            sg.Input(key="-ELEMENT-", visible=False),  # hidden controlled input
            sg.Button("", image_source=mob_type_wind_path, key="-ELEMENT-WIND-"),
            sg.Button("", image_source=mob_type_fire_path, key="-ELEMENT-FIRE-"),
            sg.Button("", image_source=mob_type_soil_path, key="-ELEMENT-SOIL-"),
            sg.Button("", image_source=mob_type_water_path, key="-ELEMENT-WATER-"),
            sg.Button(
                "", image_source=mob_type_electricity_path, key="-ELEMENT-ELECTRICITY"
            ),
        ]

        popup_window = sg.Window(
            "Add mob",
            [
                [sg.Text("Enter mob name: "), sg.Input(key="-NAME-", size=(48, 20))],
                [
                    sg.Text("Enter map name (location): "),
                    sg.Input(key="-MAP-", size=(40, 20)),
                ],
                [
                    sg.Text("Optional legacy CV name image: "),
                    sg.Input(
                        key="-IMAGE-",
                        change_submits=True,
                        size=(25, 20),
                        disabled=True,
                        text_color="#000",
                    ),
                    sg.FileBrowse(file_types=(("Image files", "*.png *.jpg *.jpeg"),)),
                ],
                [
                    sg.Text("Enter height offset: "),
                    sg.Input(key="-HEIGHT-", enable_events=True, size=(10, 20)),
                    sg.Text(
                        "(Only used by the optional CV detector)", text_color="grey"
                    ),
                ],
                [
                    sg.Text("Native species ID: "),
                    sg.Input(
                        key="-SPECIES-ID-",
                        size=(12, 20),
                        readonly=True,
                    ),
                    sg.Button("Capture targeted monster", key="-CAPTURE-SPECIES-"),
                ],
                [
                    sg.Text(
                        "Target exactly one monster in FlyFF, then capture its ID.",
                        key="-SPECIES-STATUS-",
                        text_color="grey",
                    )
                ],
                element_buttons_layout,
                [
                    sg.Frame(
                        "",
                        [[sg.Button("Reset"), sg.Button("Save")]],
                        border_width=0,
                        pad=((0, 0), (44, 0)),
                    )
                ],
            ],
            modal=True,
            size=(620, 310),
        )

        while True:
            event, values = popup_window.read()

            if event == sg.WIN_CLOSED:
                popup_window.close()
                return
            if event == "Reset":
                for elem in element_buttons_layout:
                    if elem.Disabled:
                        elem.update(disabled=False)

                for value in values:
                    if value.startswith("-"):
                        popup_window.Element(value).update("")
            if event == "-CAPTURE-SPECIES-":
                try:
                    captured = bot.capture_selected_monster()
                except Exception as error:  # noqa: BLE001 - GUI command boundary.
                    popup_window["-SPECIES-STATUS-"].update(
                        f"Capture failed: {error}",
                        text_color="red",
                    )
                else:
                    popup_window["-SPECIES-ID-"].update(str(captured.species_id))
                    popup_window["-SPECIES-STATUS-"].update(
                        f"Captured species {captured.species_id} from "
                        f"0x{captured.base_address:08X} ({captured.hp} HP).",
                        text_color="light green",
                    )
            if event == "Save":
                # Native registration needs only a name, map and captured
                # species ID. Image/height/element remain optional legacy-CV
                # metadata.
                is_form_valid = all(
                    len(values[key])
                    for key in ["-NAME-", "-MAP-", "-SPECIES-ID-"]
                )
                if values["-IMAGE-"] and not values["-HEIGHT-"]:
                    is_form_valid = False

                if is_form_valid:
                    from assets.Assets import MobInfo

                    try:
                        MobInfo.add_new_mob(
                            name=values["-NAME-"],
                            map_name=values["-MAP-"],
                            image_path=values["-IMAGE-"] or None,
                            height_offset=int(values["-HEIGHT-"] or 0),
                            element=values["-ELEMENT-"] or "unknown",
                            species_id=int(values["-SPECIES-ID-"]),
                        )
                        # Refresh the active map selection immediately so an
                        # overwritten existing mob entry gains its captured
                        # species ID without requiring a restart or map toggle.
                        self.__apply_map_selection(
                            bot,
                            self._selected_map_name,
                            publish_preview=False,
                        )
                    except Exception as error:  # noqa: BLE001 - GUI boundary.
                        popup_window["-SPECIES-STATUS-"].update(
                            f"Save failed: {error}",
                            text_color="red",
                        )
                        continue

                    popup_window.close()
                    return
                popup_window["-SPECIES-STATUS-"].update(
                    "Enter a name and map, capture the target ID, and add a "
                    "height only when a CV image is selected.",
                    text_color="red",
                )
            if event == "-HEIGHT-":
                # height validation - only numbers
                popup_window.Element(event).update(
                    re.sub("[^0-9]", "", values["-HEIGHT-"])
                )
            if isinstance(event, str) and "-ELEMENT-" in event:
                current_element = event.split("-")[2].lower()

                for elem in element_buttons_layout:
                    if elem.Disabled:
                        elem.update(disabled=False)

                popup_window.Element(event).update(disabled=True)
                popup_window.Element("-ELEMENT-").update(current_element)

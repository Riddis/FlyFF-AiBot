import difflib
import math
from threading import Lock, Thread
from time import monotonic

import cv2 as cv
import PySimpleGUI as sg
from utils.helpers import get_window_handlers, hex_variant
from runtime_bus import RuntimeBus

import re


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
        self._rl_task_running = False
        self._mapper_task_running = False
        self._active_mapper = None
        self._heading_overlay_detector = None
        self._status_mode = "Idle"
        self._last_status_message = "Ready"
        self.runtime_bus = RuntimeBus(max_logs=1500)
        self._worker_threads = set()
        self._worker_lock = Lock()
        self._last_versions = {
            "debug_frame": 0,
            "map_frame": 0,
            "video_fps": 0,
            "rl_status": 0,
            "mapper_status": 0,
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
        self.__load_settings(bot)
        while True:
            event, values = self.window.read(timeout=50)
            self.__refresh_runtime(values)

            # ACTIONS - Button events
            if event == "Exit" or event == sg.WIN_CLOSED:
                self.__shutdown(bot)
                break
            if event == "-ATTACH_WINDOW-":
                game_window_name, game_window_handler = self.__attach_window_popup()
                if game_window_name and game_window_handler:
                    bot.stop()
                    bot.setup(game_window_handler, self.runtime_bus)
                    truncated_game_window_name = (
                        game_window_name[:30] + "..." if len(game_window_name) > 30 else game_window_name
                    )
                    self.window["-ATTACHED_WINDOW-"].update(truncated_game_window_name)
                    self.__set_rl_buttons(attached=True, running=False)

            if event == "-START_BOT-":
                self.__start_rl_task(bot, mode="train")

            if event == "-RUN_AGENT-":
                self.__start_rl_task(bot, mode="agent")

            if event == "-START_MAPPER-":
                self.__start_mapper_task(bot)

            if event == "-SET_MINIMAP_ANCHOR-":
                self.__set_minimap_anchor(bot)

            if event == "-CALIBRATE_MAPPER-":
                self.__start_calibration_task(bot)

            if event == "-DEBUG_CALIBRATE_MAPPER-":
                self.__start_calibration_task(
                    bot,
                    visual_confirmation=True,
                )

            if event == "-SHOW_LOG-":
                self.__show_log_window()

            if event == "-STOP_BOT-":
                self.__set_status("Idle", "Stopped")
                if self._active_mapper is not None:
                    self._active_mapper.stop()
                bot.stop()
                self.runtime_bus.log(
                    "Stop requested. Waiting for the active worker to exit.",
                    "msg_blue",
                )

            # BOT OPTIONS - Video options
            if event == "-SHOW_FRAMES-":
                bot.set_config(show_frames=values["-SHOW_FRAMES-"])
                sg.user_settings_set_entry("-SHOW_FRAMES-", values["-SHOW_FRAMES-"])
                self.window["-SHOW_MATCHES_TEXT-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window["-SHOW_BOXES-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window["-SHOW_MARKERS-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window["-VISION_FRAME-"].update(visible=(values["-SHOW_FRAMES-"]))
                self.window.refresh()  # Combined with contents_changed, will compute the new size of the element
                self.window["-MAIN_COLUMN-"].contents_changed()
            if event == "-SHOW_MATCHES_TEXT-":
                bot.set_config(show_matches_text=values["-SHOW_MATCHES_TEXT-"])
                sg.user_settings_set_entry("-SHOW_MATCHES_TEXT-", values["-SHOW_MATCHES_TEXT-"])
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
                bot.set_config(mob_pos_match_threshold=values["-MOB_POS_MATCH_THRESHOLD-"])
                sg.user_settings_set_entry("-MOB_POS_MATCH_THRESHOLD-", values["-MOB_POS_MATCH_THRESHOLD-"])
            if event == "-MOB_STILL_ALIVE_MATCH_THRESHOLD-":
                bot.set_config(mob_still_alive_match_threshold=values["-MOB_STILL_ALIVE_MATCH_THRESHOLD-"])
                sg.user_settings_set_entry(
                    "-MOB_STILL_ALIVE_MATCH_THRESHOLD-", values["-MOB_STILL_ALIVE_MATCH_THRESHOLD-"]
                )
            if event == "-MOB_EXISTENCE_MATCH_THRESHOLD-":
                bot.set_config(mob_existence_match_threshold=values["-MOB_EXISTENCE_MATCH_THRESHOLD-"])
                sg.user_settings_set_entry("-MOB_EXISTENCE_MATCH_THRESHOLD-", values["-MOB_EXISTENCE_MATCH_THRESHOLD-"])
            if event == "-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-":
                bot.set_config(
                    inventory_perin_converter_match_threshold=values["-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-"]
                )
                sg.user_settings_set_entry(
                    "-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-", values["-INVENTORY_PERIN_CONVERTER_MATCH_THRESHOLD-"]
                )
            if event == "-INVENTORY_ICONS_MATCH_THRESHOLD-":
                bot.set_config(inventory_icons_match_threshold=values["-INVENTORY_ICONS_MATCH_THRESHOLD-"])
                sg.user_settings_set_entry(
                    "-INVENTORY_ICONS_MATCH_THRESHOLD-", values["-INVENTORY_ICONS_MATCH_THRESHOLD-"]
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
                        sg.user_settings_set_entry("-MOBS_KILL_GOAL-", values["-MOBS_KILL_GOAL-"])
                    except ValueError:
                        sg.cprint("Invalid mobs kill goal")
                        self.window["-MOBS_KILL_GOAL-"].update("infinite")
                        bot.set_config(mobs_kill_goal=None)
                        sg.user_settings_set_entry("-MOBS_KILL_GOAL-", "infinite")
            if event == "-FIGHT_TIME_LIMIT_SEC-":
                try:
                    fight_time_limit_sec = int(values["-FIGHT_TIME_LIMIT_SEC-"])
                    bot.set_config(fight_time_limit_sec=fight_time_limit_sec)
                    sg.user_settings_set_entry("-FIGHT_TIME_LIMIT_SEC-", values["-FIGHT_TIME_LIMIT_SEC-"])
                except ValueError:
                    sg.cprint("Invalid fight time limit")
                    self.window["-FIGHT_TIME_LIMIT_SEC-"].update("8")
                    bot.set_config(fight_time_limit_sec=8)
                    sg.user_settings_set_entry("-FIGHT_TIME_LIMIT_SEC-", "8")
            if event == "-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-":
                try:
                    delay_to_check_mob_still_alive_sec = float(values["-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-"])
                    bot.set_config(delay_to_check_mob_still_alive_sec=delay_to_check_mob_still_alive_sec)
                    sg.user_settings_set_entry(
                        "-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-", values["-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-"]
                    )
                except ValueError:
                    sg.cprint("Invalid delay to check if mob is still alive")
                    self.window["-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-"].update("0.25")
                    bot.set_config(delay_to_check_mob_still_alive_sec=0.25)
                    sg.user_settings_set_entry("-DELAY_TO_CHECK_MOB_STILL_ALIVE_SEC-", "0.25")
            if event == "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-":
                try:
                    convert_penya_to_perins_timer_min = float(values["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"])
                    bot.set_config(convert_penya_to_perins_timer_min=convert_penya_to_perins_timer_min)
                    sg.user_settings_set_entry(
                        "-CONVERT_PENYA_TO_PERINS_TIMER_MIN-", values["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"]
                    )
                except ValueError:
                    sg.cprint("Invalid convert penya to perins timer, must be in minutes")
                    self.window["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"].update("30")
                    bot.set_config(convert_penya_to_perins_timer_min=30)
                    sg.user_settings_set_entry("-CONVERT_PENYA_TO_PERINS_TIMER_MIN-", "30")

            # MOBS - Mobs configuration
            if event == "-SELECT_MOBS-":
                self.__select_mobs_popup(bot)
            
            if event == "-ADD_MOB-":
                self.__add_mob_popup()

            if event == "-DELETE_MOB-":
                self.__select_mobs_popup(bot, is_delete_form=True)


    def __start_worker(self, *, target, args, name):
        def guarded_target():
            try:
                target(*args)
            finally:
                with self._worker_lock:
                    self._worker_threads.discard(thread)

        thread = Thread(
            target=guarded_target,
            daemon=False,
            name=name,
        )
        with self._worker_lock:
            self._worker_threads.add(thread)
        thread.start()

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
            and now - self._last_preview_render_at
            >= self._preview_render_interval
        ):
            self._last_versions["debug_frame"] = version
            image = self.__draw_heading_overlay(frame.copy())
            resolution = values.get("-DEBUG_IMG_WIDTH-", "960x540")
            width, height = self.frame_resolutions[resolution]
            image = self.__fit_image(image, width, height)
            encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, 82])
            if encoded[0]:
                self.window["-DEBUG_IMAGE-"].update(
                    data=encoded[1].tobytes()
                )
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
                self.window["-MAPPER_IMAGE-"].update(
                    data=encoded[1].tobytes()
                )
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
        ):
            version, status = self.runtime_bus.read_latest(
                key,
                self._last_versions[key],
            )
            if status is not None:
                self._last_versions[key] = version
                self.__set_status(mode, status)

        # Bounded log draining at <=5 FPS.
        if now - self._last_log_render_at >= 0.20:
            for level, message in self.runtime_bus.drain_logs(80):
                color = self.logger_events_color.get(
                    level,
                    ("white", "black"),
                )
                sg.cprint(message, c=color)
                self._last_status_message = message
            self._last_log_render_at = now

        for completion in self.runtime_bus.drain_completions():
            task = completion.worker_name
            result = completion.result
            if task == "rl":
                self._rl_task_running = False
            else:
                self._mapper_task_running = False
                self._active_mapper = None

            self.__set_rl_buttons(attached=True, running=False)
            if result.get("ok"):
                self.runtime_bus.log(
                    result.get("message", f"{task} finished."),
                    "msg_green",
                )
            else:
                self.runtime_bus.log(
                    f"{task.title()} failed: "
                    f"{result.get('error', 'unknown error')}",
                    "msg_red",
                )

        request = self.runtime_bus.pop_confirmation()
        if request is not None:
            request.result = self.__confirm_heading_reading(
                request.frame,
                request.angle_deg,
                request.confidence,
                request.context,
            )
            request.completed.set()

    def __shutdown(self, bot):
        self.__set_status("Stopping", "Stopping workers safely…")
        if self._active_mapper is not None:
            try:
                self._active_mapper.stop()
            except Exception:
                pass
        try:
            bot.stop()
        except Exception:
            pass

        deadline = monotonic() + 5.0
        while monotonic() < deadline:
            with self._worker_lock:
                workers = [
                    thread
                    for thread in self._worker_threads
                    if thread.is_alive()
                ]
            if not workers:
                break
            for thread in workers:
                thread.join(timeout=0.10)
            try:
                self.window.refresh()
            except Exception:
                break

        try:
            bot.close()
        except Exception:
            pass
        self.runtime_bus.close()

    def __set_rl_buttons(self, attached, running):
        """Keep the RL action buttons in a consistent state."""
        self.window["-START_BOT-"].update(
            disabled=(not attached or running)
        )
        self.window["-RUN_AGENT-"].update(
            disabled=(not attached or running)
        )
        self.window["-START_MAPPER-"].update(
            disabled=(not attached or running)
        )
        self.window["-SET_MINIMAP_ANCHOR-"].update(
            disabled=(not attached or running)
        )
        self.window["-CALIBRATE_MAPPER-"].update(
            disabled=(not attached or running)
        )
        self.window["-DEBUG_CALIBRATE_MAPPER-"].update(
            disabled=(not attached or running)
        )
        self.window["-STOP_BOT-"].update(
            disabled=(not attached or not running)
        )

    def __start_rl_task(self, bot, mode):
        if self._rl_task_running:
            sg.cprint("An RL task is already running.", c=("white", "red"))
            return

        if not bot.config.get("selected_mobs"):
            sg.cprint(
                "Select at least one mob before starting.",
                c=("white", "red"),
            )
            return

        self._rl_task_running = True
        self.__set_rl_buttons(attached=True, running=True)

        label = "training" if mode == "train" else "trained agent"
        self.__set_status(
            "Training" if mode == "train" else "Agent",
            f"Starting {label}...",
        )
        self.runtime_bus.log(f"Starting {label}...", "msg_blue")

        self.__start_worker(
            target=self.__run_rl_task,
            args=(bot, mode),
            name=f"flyff-rl-{mode}",
        )

    def __run_rl_task(self, bot, mode):
        def report(message):
            self.runtime_bus.publish_latest("rl_status", str(message))
            self.runtime_bus.log(str(message), "msg_blue")
            self.runtime_bus.heartbeat("rl")

        try:
            from train import run_trained_agent, train_agent

            if mode == "train":
                model_path = train_agent(
                    bot,
                    status_callback=report,
                )
                result = {
                    "ok": True,
                    "message": f"Training finished. Model saved to {model_path}.",
                }
            elif mode == "agent":
                run_trained_agent(
                    bot,
                    status_callback=report,
                )
                result = {
                    "ok": True,
                    "message": "Trained agent stopped.",
                }
            else:
                raise ValueError(f"Unknown RL mode: {mode}")

        except Exception as error:
            result = {"ok": False, "error": str(error)}
        finally:
            try:
                bot.stop_movement()
            except Exception:
                pass
            bot.stop()

        self.runtime_bus.complete("rl", result)

    def __start_mapper_task(self, bot):
        if self._rl_task_running or self._mapper_task_running:
            sg.cprint(
                "Another control task is already running.",
                c=("white", "red"),
            )
            return

        if not bot.is_ready:
            sg.cprint(
                "Attach the Flyff window first.",
                c=("white", "red"),
            )
            return

        self._mapper_task_running = True
        self.__set_rl_buttons(attached=True, running=True)
        self.__set_status("Mapping", "Starting autonomous mapper...")
        self.__clear_live_map("Mapping is starting…")
        self.runtime_bus.log(
            "Starting autonomous mapper...",
            "msg_blue",
        )

        self.__start_worker(
            target=self.__run_mapper_task,
            args=(bot,),
            name="flyff-mapper",
        )

    def __run_mapper_task(self, bot):
        def report(message):
            self.runtime_bus.publish_latest("mapper_status", str(message))
            self.runtime_bus.log(str(message), "msg_blue")
            self.runtime_bus.heartbeat("mapper")

        def publish_map(frame):
            self.runtime_bus.publish_latest("map_frame", frame)
            self.runtime_bus.heartbeat("mapper")

        try:
            from mapper import Mapper

            mapper = Mapper(
                bot,
                status_callback=report,
                frame_callback=publish_map,
            )
            self._active_mapper = mapper
            output_dir = mapper.run()
            result = {
                "ok": True,
                "message": f"Mapper stopped. Map saved to {output_dir}.",
            }
        except Exception as error:
            result = {"ok": False, "error": str(error)}
        finally:
            self._active_mapper = None
            try:
                bot.stop_movement()
            except Exception:
                pass

        self.runtime_bus.complete("mapper", result)

    def __set_minimap_anchor(self, bot):
        if self._rl_task_running or self._mapper_task_running:
            sg.cprint(
                "Stop the active control task before setting the minimap "
                "center.",
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
            from mapper import MinimapAnchorSetup

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
        except Exception as error:
            sg.cprint(
                f"Minimap-center setup failed: {error}",
                c=("white", "red"),
            )

    def __clear_manual_calibration_artifacts(self):
        """
        Manual recalibration starts a new diagnostic session.

        Preserve minimap_anchor.json because that is fixed UI geometry, but
        remove prior timing calibration and old heading-debug captures.
        """
        from pathlib import Path
        import shutil

        project_root = Path(__file__).resolve().parent
        calibration_files = [
            project_root / "mapper" / "calibration.json",
        ]
        debug_directories = [
            project_root / "debug" / "minimap_heading",
        ]

        removed = []
        for file_path in calibration_files:
            if file_path.exists():
                file_path.unlink()
                removed.append(str(file_path))

        for directory in debug_directories:
            if directory.exists():
                shutil.rmtree(directory)
                removed.append(str(directory))
            directory.mkdir(parents=True, exist_ok=True)

        if removed:
            sg.cprint(
                "Manual recalibration cleared the previous calibration and "
                "minimap-heading debug files.",
                c=("white", "blue"),
            )
        else:
            sg.cprint(
                "Manual recalibration started with no old calibration/debug "
                "files to clear.",
                c=("white", "blue"),
            )

    def __start_calibration_task(
        self,
        bot,
        visual_confirmation=False,
    ):
        if self._rl_task_running or self._mapper_task_running:
            sg.cprint(
                "Another control task is already running.",
                c=("white", "red"),
            )
            return

        if not bot.is_ready:
            sg.cprint(
                "Attach the Flyff window first.",
                c=("white", "red"),
            )
            return

        self._mapper_task_running = True
        self.__set_rl_buttons(attached=True, running=True)
        self.__clear_manual_calibration_artifacts()

        self.__set_status(
            "Calibration",
            (
                "Starting visual-confirmation calibration..."
                if visual_confirmation
                else "Starting rotation calibration..."
            ),
        )
        self.__clear_live_map(
            "No map yet — start Map Area to build the occupancy preview."
        )
        self.runtime_bus.log(
            "Starting rotation calibration...",
            "msg_blue",
        )

        self.__start_worker(
            target=self.__run_calibration_task,
            args=(bot, visual_confirmation),
            name="flyff-mapper-calibration",
        )

    def __run_calibration_task(
        self,
        bot,
        visual_confirmation=False,
    ):
        def report(message):
            self.runtime_bus.publish_latest(
                "mapper_status",
                str(message),
            )
            self.runtime_bus.log(str(message), "msg_blue")
            self.runtime_bus.heartbeat("calibration")

        try:
            from mapper import RotationCalibrator

            calibrator = RotationCalibrator(
                bot,
                status_callback=report,
                frame_callback=None,
                visual_confirmation_callback=(
                    self.runtime_bus.request_heading_confirmation
                    if visual_confirmation
                    else None
                ),
            )
            self._active_mapper = calibrator
            calibration_path = calibrator.run(manual=True)
            result = {
                "ok": True,
                "message": (
                    "Calibration complete. Saved to "
                    f"{calibration_path}."
                ),
            }
        except Exception as error:
            result = {"ok": False, "error": str(error)}
        finally:
            self._active_mapper = None
            try:
                bot.stop_movement()
            except Exception:
                pass

        self.runtime_bus.complete("calibration", result)

    def close(self):
        self.runtime_bus.close()
        self.window.close()

    def __load_settings(self, bot):
        show_frames = sg.user_settings_get_entry("-SHOW_FRAMES-", True)
        self.window["-SHOW_FRAMES-"].update(show_frames)
        self.window["-SHOW_MATCHES_TEXT-"].update(visible=show_frames)
        self.window["-SHOW_BOXES-"].update(visible=show_frames)
        self.window["-SHOW_MARKERS-"].update(visible=show_frames)
        self.window["-VISION_FRAME-"].update(visible=show_frames)
        bot.set_config(show_frames=show_frames)

        show_matches_text = sg.user_settings_get_entry(
            "-SHOW_MATCHES_TEXT-", False
        )
        self.window["-SHOW_MATCHES_TEXT-"].update(show_matches_text)
        bot.set_config(show_matches_text=show_matches_text)

        show_mobs_pos_boxes = sg.user_settings_get_entry(
            "-SHOW_BOXES-", False
        )
        self.window["-SHOW_BOXES-"].update(show_mobs_pos_boxes)
        bot.set_config(show_mobs_pos_boxes=show_mobs_pos_boxes)

        show_mobs_pos_markers = sg.user_settings_get_entry(
            "-SHOW_MARKERS-", True
        )
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
        self.window["-CONVERT_PENYA_TO_PERINS_TIMER_MIN-"].update(
            convert_timer
        )
        bot.set_config(convert_penya_to_perins_timer_min=convert_timer)

        all_mobs = bot.get_all_mobs()
        selected_names = sg.user_settings_get_entry(
            "saved_selected_mobs", []
        )
        selected_names = [
            name for name in selected_names if name in all_mobs
        ]
        selected = [
            all_mobs[key]
            for key in all_mobs
            if key in selected_names
        ]
        bot.set_config(selected_mobs=selected)

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
                        "Set Minimap Center",
                        disabled=True,
                        key="-SET_MINIMAP_ANCHOR-",
                        expand_x=True,
                    ),
                    sg.Button(
                        "Calibrate Mapper",
                        disabled=True,
                        key="-CALIBRATE_MAPPER-",
                        expand_x=True,
                    ),
                ],
                [
                    sg.Button(
                        "Debug Heading Calibration",
                        disabled=True,
                        key="-DEBUG_CALIBRATE_MAPPER-",
                        expand_x=True,
                    ),
                ],
                [
                    sg.Button(
                        "Map Area",
                        disabled=True,
                        key="-START_MAPPER-",
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
                        default_value="960x540",
                        readonly=True,
                        key="-DEBUG_IMG_WIDTH-",
                    ),
                ],
                [
                    sg.Image(
                        filename="",
                        key="-DEBUG_IMAGE-",
                        size=(960, 540),
                    )
                ],
            ],
            visible=True,
            expand_x=True,
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
                    sg.Text("Pang", text_color="cyan"),
                    sg.Text("Player + heading", text_color="yellow"),
                ],
                [
                    sg.Image(
                        data=self.__make_live_map_placeholder(
                            "No map yet — click Map Area to begin."
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

    def __fit_image(self, image, target_width, target_height):
        """Letterbox an image without changing its aspect ratio."""
        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]
        scale = min(
            target_width / max(1, width),
            target_height / max(1, height),
        )
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv.resize(
            image,
            (resized_width, resized_height),
            interpolation=(
                cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
            ),
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

    def __draw_heading_overlay(self, image):
        """Draw the live minimap heading marker on Bot Vision."""
        try:
            if self._heading_overlay_detector is None:
                from mapper import MinimapHeadingDetector
                self._heading_overlay_detector = MinimapHeadingDetector()

            reading = self._heading_overlay_detector.read_fast(image)
            if reading is None:
                return image

            center_x, center_y = reading.center
            angle = math.radians(reading.angle_deg)
            length = 44
            end_x = int(round(center_x + math.sin(angle) * length))
            end_y = int(round(center_y - math.cos(angle) * length))

            color = (0, 215, 255)
            if reading.is_stale:
                color = (0, 140, 255)

            cv.circle(image, (center_x, center_y), 6, color, 2)
            cv.arrowedLine(
                image,
                (center_x, center_y),
                (end_x, end_y),
                color,
                3,
                tipLength=0.28,
            )
            cv.putText(
                image,
                f"{reading.angle_deg:.0f} deg {reading.confidence:.2f}",
                (max(4, center_x - 78), max(18, center_y - 18)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv.LINE_AA,
            )
        except Exception:
            return image

        return image

    def __set_status(self, mode, message):
        self._status_mode = mode
        self._last_status_message = str(message)
        if hasattr(self, "window"):
            self.window["-STATUS_MODE-"].update(f"Mode: {mode}")
            self.window["-STATUS_MESSAGE-"].update(
                self._last_status_message[-240:]
            )

    def __show_log_window(self):
        try:
            log_text = self.window["-ML-"].get()
        except Exception:
            log_text = ""

        log_window = sg.Window(
            "Flyff FVF Log",
            [
                [
                    sg.Multiline(
                        log_text,
                        size=(110, 35),
                        disabled=True,
                        autoscroll=True,
                    )
                ],
                [sg.Button("Close")],
            ],
            modal=True,
            resizable=True,
            finalize=True,
        )
        while True:
            event, _ = log_window.read()
            if event in (sg.WIN_CLOSED, "Close"):
                break
        log_window.close()

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
                int(round(center[0] + math.sin(radians) * 70)),
                int(round(center[1] - math.cos(radians) * 70)),
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
        except Exception:
            pass

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
                        "Does the yellow arrow match the character's actual "
                        "heading?"
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
        handlers = get_window_handlers()
        popup_window = sg.Window(
            "Attach Window",
            [
                [sg.Text("Please select the window to attach to:")],
                [sg.DropDown(list(handlers.keys()), key="-DROP-"), sg.Button("Refresh")],
                [sg.OK(), sg.Cancel()],
            ],
            size=(400, 100),
        )
        while True:
            event, values = popup_window.read()
            if event == "Refresh":
                handlers = get_window_handlers()
                popup_window["-DROP-"].update(values=list(handlers.keys()))
            if event in (sg.WIN_CLOSED, "Cancel"):
                popup_window.close()
                return None, None
            if event == "OK":
                popup_window.close()
                return values["-DROP-"], handlers[values["-DROP-"]]
            
    def __select_mobs_popup(self, bot, is_delete_form=False):
        all_mobs = bot.get_all_mobs()
        selected_mobs_names = [mob["name"] for mob in bot.config["selected_mobs"] if mob["name"] in all_mobs]

        all_mobs_titles = [f"{name} - {params['element']} - {params['map_name']}" for (name, params) in dict.items(all_mobs)]
        selected_mobs_titles = [
            f"{name} - {params['element']} - {params['map_name']}" for (name, params) in dict.items(all_mobs)
            if name in selected_mobs_names
        ]
        if is_delete_form: selected_mobs_titles = []
        last_highlighted_mob = None

        popup_window = sg.Window(
            "Select Mobs" if not is_delete_form else "Delete mobs",
            [
                [sg.Text(f"Select the mobs to {'kill' if not is_delete_form else 'delete'}:")],
                [sg.Text("Find: "), sg.Input(enable_events=True, expand_x=True, key="-MOBS_SEARCH-")],
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
                [sg.Button("Reset"), sg.Button("Save" if not is_delete_form else "Delete", button_color=None if not is_delete_form else "#ea4335")],
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
                best_match = difflib.get_close_matches(search, all_mobs_titles, n=1, cutoff=0.0)
                if last_highlighted_mob is not None:
                    listbox.Widget.itemconfigure(last_highlighted_mob, bg=listbox.BackgroundColor)
                    last_highlighted_mob = None
                if len(best_match) > 0:
                    best_match_index = all_mobs_titles.index(best_match[0])
                    listbox.Widget.itemconfigure(best_match_index, bg=hex_variant(listbox.BackgroundColor, -20))
                    listbox.update(scroll_to_index=best_match_index)
                    last_highlighted_mob = best_match_index
            else:
                if last_highlighted_mob is not None:
                    listbox.Widget.itemconfigure(last_highlighted_mob, bg=listbox.BackgroundColor)
                    last_highlighted_mob = None

            if event == "-MOBS_LIST-" and len(values["-MOBS_LIST-"]):
                selected_mobs_indexes = [all_mobs_titles.index(mob) for mob in values["-MOBS_LIST-"]]
                listbox.update(set_to_index=selected_mobs_indexes)

            if event == "Reset":
                listbox.update(set_to_index=[])
            if event == "Save":
                selected_mobs_indexes = [all_mobs_titles.index(mob) for mob in values["-MOBS_LIST-"]]
                popup_window.close()
                all_names = list(dict.keys(all_mobs))
                selected_names = [all_names[i] for i in selected_mobs_indexes]
                sg.user_settings_set_entry("saved_selected_mobs", selected_names)
                bot.set_config(selected_mobs=[all_mobs[name] for name in selected_names])
                return
            if event == "Delete":
                from assets.Assets import MobInfo
                deleted_mobs_names = [mob.split("-")[0].strip() for mob in values["-MOBS_LIST-"]]
                popup_window.close()
                # unselect deleted mobs if they were selected
                bot.set_config(selected_mobs=[all_mobs[name] for name in selected_mobs_names if name not in deleted_mobs_names])
                MobInfo.delete_mobs(deleted_mobs_names)
                return
    
    def __add_mob_popup(self):
        from assets.Assets import mob_type_wind_path, mob_type_fire_path, mob_type_soil_path, mob_type_water_path, mob_type_electricity_path

        element_buttons_layout = [
            sg.Text("Select mob element: "),
            sg.Input(key="-ELEMENT-", visible=False), # hidden controlled input
            sg.Button("", image_source=mob_type_wind_path, key="-ELEMENT-WIND-"),
            sg.Button("", image_source=mob_type_fire_path, key="-ELEMENT-FIRE-"),
            sg.Button("", image_source=mob_type_soil_path, key="-ELEMENT-SOIL-"),
            sg.Button("", image_source=mob_type_water_path, key="-ELEMENT-WATER-"),
            sg.Button("", image_source=mob_type_electricity_path, key="-ELEMENT-ELECTRICITY")
        ]

        popup_window = sg.Window(
            "Add mob",
            [
                [sg.Text("Enter mob name: "), sg.Input(key="-NAME-", size=(48,20))],
                [sg.Text("Enter map name (location): "), sg.Input(key="-MAP-", size=(40,20))],
                [
                    sg.Text("Choose an image file (mob name): "),
                    sg.Input(key="-IMAGE-", change_submits=True, size=(25, 20), disabled=True, text_color="#000"),
                    sg.FileBrowse(file_types=(("Image files", "*.png *.jpg *.jpeg"),))
                ],
                [sg.Text("Enter height offset: "), sg.Input(key="-HEIGHT-", enable_events=True, size=(10,20)), sg.Text("(Number, usually in range from 40 to 100)", text_color="grey")],
                element_buttons_layout,
                [sg.Frame("", [[sg.Button("Reset"), sg.Button("Save")]], border_width=0, pad=((0, 0),(44, 0)))],
            ],
            modal=True,
            size=(500, 225)
        )

        while True:
            event, values = popup_window.read()

            if event == sg.WIN_CLOSED:
                popup_window.close()
                return
            if event == "Reset":
                for elem in element_buttons_layout:
                     if elem.Disabled: elem.update(disabled=False)

                for value in values:
                    if value.startswith("-"):
                        popup_window.Element(value).update("")
                pass
            if event == "Save":
                # form validation
                is_form_valid = True
                for key in ["-NAME-", "-MAP-", "-IMAGE-", "-HEIGHT-", "-ELEMENT-"]:
                    if not len(values[key]):
                        is_form_valid = False
                        break

                if is_form_valid:
                    from assets.Assets import MobInfo
                    MobInfo.add_new_mob(name=values["-NAME-"], map_name=values["-MAP-"], image_path=values["-IMAGE-"],
                                        height_offset=int(values["-HEIGHT-"]), element=values["-ELEMENT-"])
                    popup_window.close()
                    return
            if event == "-HEIGHT-":
                # height validation - only numbers
                popup_window.Element(event).update(re.sub("[^0-9]","", values["-HEIGHT-"]))
                pass
            if isinstance(event, str) and "-ELEMENT-" in event:
                current_element = event.split("-")[2].lower()

                for elem in element_buttons_layout:
                     if elem.Disabled: elem.update(disabled=False)

                popup_window.Element(event).update(disabled=True)
                popup_window.Element("-ELEMENT-").update(current_element)

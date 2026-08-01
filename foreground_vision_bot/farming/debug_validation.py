from __future__ import annotations

import json
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


@dataclass(frozen=True, slots=True)
class ValidationArtifacts:
    session_directory: Path
    events_path: Path
    summary_path: Path
    archive_path: Path


class TrainingDataValidationRecorder:
    """Write self-contained evidence for one live farming-data validation run."""

    def __init__(
        self,
        base_directory: str | Path,
        *,
        frame_provider: Callable[[], np.ndarray | None] | None = None,
        maximum_screenshots: int = 16,
    ) -> None:
        if maximum_screenshots < 0:
            raise ValueError("maximum_screenshots cannot be negative")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        root = Path(base_directory)
        self.session_directory = root / f"validation-{timestamp}"
        self.frames_directory = self.session_directory / "frames"
        self.events_path = self.session_directory / "events.jsonl"
        self.summary_path = self.session_directory / "summary.json"
        self.readme_path = self.session_directory / "README.txt"
        self.archive_path = root / f"SEND_TO_CHATGPT_training_data_debug_{timestamp}.zip"
        self.frames_directory.mkdir(parents=True, exist_ok=True)
        self._events_handle = self.events_path.open("x", encoding="utf-8")
        self._frame_provider = frame_provider
        self._maximum_screenshots = int(maximum_screenshots)
        self._screenshots = 0
        self._screenshot_reasons: set[str] = set()
        self._event_count = 0
        self._step_count = 0
        self._action_counts: Counter[str] = Counter()
        self._ocr_outcomes: Counter[str] = Counter()
        self._ocr_baseline: int | None = None
        self._ocr_latest: int | None = None
        self._ocr_positive_delta = 0
        self._native_kills = 0
        self._casts = 0
        self._casts_with_candidates = 0
        self._cast_candidate_total = 0
        self._confirmed_base_addresses: set[int] = set()
        self._kill_poll_successful_reads = 0
        self._kill_poll_failed_reads = 0
        self._candidate_traces = 0
        self._candidate_always_present = 0
        self._candidate_single_absence = 0
        self._candidate_confirmed_traces = 0
        self._actor_samples = 0
        self._maximum_visible_actors = 0
        self._minimum_visible_actors: int | None = None
        self._observed_species: Counter[int] = Counter()
        self._nonfinite_observations = 0
        self._observation_sizes: Counter[int] = Counter()
        self._movement_steps = 0
        self._movement_steps_with_displacement = 0
        self._contacts = 0
        self._hp_decreases = 0
        self._hp_zero_transitions = 0
        self._candidate_disappearances = 0
        self._pointer_modes: Counter[str] = Counter()
        self._pointer_generations: Counter[int] = Counter()
        self._errors: list[str] = []
        self._finished = False

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(key): TrainingDataValidationRecorder._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [TrainingDataValidationRecorder._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _actor_map(world: object) -> dict[int, Mapping[str, object]]:
        if not isinstance(world, Mapping):
            return {}
        actors = world.get("actors")
        if not isinstance(actors, list):
            return {}
        result: dict[int, Mapping[str, object]] = {}
        for actor in actors:
            if not isinstance(actor, Mapping):
                continue
            try:
                base = int(actor.get("base", 0))
            except (TypeError, ValueError):
                continue
            if base > 0:
                result[base] = actor
        return result

    def _capture_frame(self, reason: str) -> None:
        if (
            self._frame_provider is None
            or self._screenshots >= self._maximum_screenshots
            or reason in self._screenshot_reasons
        ):
            return
        try:
            frame = self._frame_provider()
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return
            path = self.frames_directory / f"{self._screenshots:02d}_{reason}.png"
            if cv.imwrite(str(path), frame):
                self._screenshots += 1
                self._screenshot_reasons.add(reason)
        except Exception as error:  # noqa: BLE001 - diagnostics must not stop control.
            self._errors.append(f"screenshot {reason}: {type(error).__name__}: {error}")

    def record(self, raw_event: Mapping[str, object]) -> None:
        if self._finished:
            return
        event = self._json_safe(dict(raw_event))
        self._events_handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        self._events_handle.flush()
        self._event_count += 1

        event_name = str(event.get("event", "unknown"))
        if event_name == "reset":
            self._capture_frame("reset")
        if event_name != "step":
            return

        self._step_count += 1
        action_name = str(event.get("action_name", "UNKNOWN"))
        self._action_counts[action_name] += 1
        if action_name.startswith("RUN_"):
            self._movement_steps += 1

        info = event.get("info")
        if not isinstance(info, Mapping):
            info = {}
        try:
            native_delta = int(info.get("native_kill_delta", 0))
        except (TypeError, ValueError):
            native_delta = 0
        self._native_kills += max(0, native_delta)
        if native_delta > 0:
            self._capture_frame("native_kill")

        try:
            displacement = float(info.get("player_displacement_cells", 0.0))
        except (TypeError, ValueError):
            displacement = 0.0
        if action_name.startswith("RUN_") and displacement > 0.05:
            self._movement_steps_with_displacement += 1
        if bool(info.get("contact", False)):
            self._contacts += 1

        ocr = event.get("ocr")
        if isinstance(ocr, Mapping):
            outcome = str(ocr.get("outcome", "unknown"))
            self._ocr_outcomes[outcome] += 1
            value = ocr.get("value")
            if isinstance(value, int) and not isinstance(value, bool):
                if self._ocr_baseline is None:
                    self._ocr_baseline = value
                self._ocr_latest = value
            delta = ocr.get("delta")
            if isinstance(delta, int) and not isinstance(delta, bool) and delta > 0:
                self._ocr_positive_delta += delta
                self._capture_frame("ocr_increment")
            if outcome != "ok":
                self._capture_frame(f"ocr_{outcome}")

        candidates = event.get("cast_candidates")
        if action_name == "CAST_EVA":
            self._casts += 1
            candidate_count = len(candidates) if isinstance(candidates, list) else 0
            self._cast_candidate_total += candidate_count
            if candidate_count > 0:
                self._casts_with_candidates += 1
                self._capture_frame("cast_with_candidates")
            else:
                self._capture_frame("cast_without_candidates")

        kill_result = event.get("kill_result")
        if isinstance(kill_result, Mapping):
            try:
                self._kill_poll_successful_reads += max(
                    0, int(kill_result.get("successful_reads", 0))
                )
                self._kill_poll_failed_reads += max(
                    0, int(kill_result.get("failed_reads", 0))
                )
            except (TypeError, ValueError):
                pass
            confirmed = kill_result.get("confirmed")
            if isinstance(confirmed, list):
                for item in confirmed:
                    if isinstance(item, Mapping):
                        try:
                            self._confirmed_base_addresses.add(int(item.get("base", 0)))
                        except (TypeError, ValueError):
                            pass
            candidate_diagnostics = kill_result.get("candidate_diagnostics")
            if isinstance(candidate_diagnostics, list):
                for item in candidate_diagnostics:
                    if not isinstance(item, Mapping):
                        continue
                    self._candidate_traces += 1
                    try:
                        maximum_absence = int(
                            item.get("maximum_consecutive_absence", 0)
                        )
                    except (TypeError, ValueError):
                        maximum_absence = 0
                    if maximum_absence <= 0:
                        self._candidate_always_present += 1
                    elif maximum_absence == 1:
                        self._candidate_single_absence += 1
                    if bool(item.get("confirmed", False)):
                        self._candidate_confirmed_traces += 1

        before = event.get("before")
        after = event.get("after")
        before_actors = self._actor_map(before)
        after_actors = self._actor_map(after)
        self._actor_samples += len(after_actors)
        visible = len(after_actors)
        self._maximum_visible_actors = max(self._maximum_visible_actors, visible)
        self._minimum_visible_actors = visible if self._minimum_visible_actors is None else min(self._minimum_visible_actors, visible)
        for actor in after_actors.values():
            try:
                self._observed_species[int(actor.get("species", 0))] += 1
            except (TypeError, ValueError):
                continue
        for base, before_actor in before_actors.items():
            after_actor = after_actors.get(base)
            if after_actor is None:
                continue
            try:
                before_hp = int(before_actor.get("hp", 0))
                after_hp = int(after_actor.get("hp", 0))
            except (TypeError, ValueError):
                continue
            if after_hp < before_hp:
                self._hp_decreases += 1
            if before_hp > 0 and after_hp == 0:
                self._hp_zero_transitions += 1
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, Mapping):
                    continue
                try:
                    base = int(item.get("base", 0))
                except (TypeError, ValueError):
                    continue
                if base in before_actors and base not in after_actors:
                    self._candidate_disappearances += 1

        if isinstance(after, Mapping):
            pointer = after.get("pointer")
            if isinstance(pointer, Mapping):
                self._pointer_modes[str(pointer.get("mode", "unknown"))] += 1
                try:
                    self._pointer_generations[int(pointer.get("generation", -1))] += 1
                except (TypeError, ValueError):
                    pass

        observation = event.get("observation")
        if isinstance(observation, Mapping):
            try:
                size = int(observation.get("size", 0))
            except (TypeError, ValueError):
                size = 0
            self._observation_sizes[size] += 1
            if not bool(observation.get("finite", False)):
                self._nonfinite_observations += 1

    def note_error(self, error: BaseException) -> None:
        self._errors.append(f"{type(error).__name__}: {error}")

    def _build_checks(self) -> tuple[list[dict[str, object]], str, list[str]]:
        checks: list[dict[str, object]] = []
        recommendations: list[str] = []

        def add(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})

        if self._step_count <= 0:
            add("runtime_steps", "fail", "No environment steps were recorded.")
        else:
            add("runtime_steps", "pass", f"Recorded {self._step_count} live steps.")

        if self._nonfinite_observations:
            add("observation_finiteness", "fail", f"{self._nonfinite_observations} observations contained NaN/Inf values.")
            recommendations.append("Inspect observation construction before any further training.")
        else:
            add("observation_finiteness", "pass", "Every recorded model observation was finite.")

        if len(self._observation_sizes) != 1 or 0 in self._observation_sizes:
            add("observation_shape", "fail", f"Observed sizes: {dict(self._observation_sizes)}")
        else:
            size = next(iter(self._observation_sizes))
            add("observation_shape", "pass", f"Observation size stayed fixed at {size}.")

        if self._actor_samples <= 0:
            add("native_actor_input", "fail", "No selected monsters were ever visible to the native reader.")
            recommendations.append("Verify selected species and actor-cache coverage.")
        else:
            add("native_actor_input", "pass", f"Observed {self._actor_samples} actor samples; peak visible={self._maximum_visible_actors}.")

        if self._movement_steps and self._movement_steps_with_displacement == 0:
            add("player_movement", "fail", "Movement actions were sent but native coordinates never changed.")
            recommendations.append("Check focus/input and recovered player coordinate fields.")
        elif self._movement_steps:
            ratio = self._movement_steps_with_displacement / self._movement_steps
            status = "pass" if ratio >= 0.25 else "warn"
            add("player_movement", status, f"Movement observed on {self._movement_steps_with_displacement}/{self._movement_steps} movement steps.")
        else:
            add("player_movement", "warn", "No movement actions were recorded.")

        ocr_total = sum(self._ocr_outcomes.values())
        ocr_ok = self._ocr_outcomes.get("ok", 0)
        if ocr_total == 0:
            add("kill_counter_ocr", "fail", "The kill-counter OCR was never sampled.")
            recommendations.append("Check frame capture and the kill-counter reader call path.")
        else:
            success_ratio = ocr_ok / ocr_total
            status = "pass" if success_ratio >= 0.80 else "warn" if success_ratio >= 0.50 else "fail"
            add("kill_counter_ocr", status, f"OCR success={ocr_ok}/{ocr_total} ({success_ratio:.1%}); outcomes={dict(self._ocr_outcomes)}.")
            if status != "pass":
                recommendations.append("Keep the kill/Penya tracker fully visible and inspect the included OCR screenshots.")

        if self._casts_with_candidates <= 0:
            add("eva_candidate_input", "inconclusive", f"EVA casts={self._casts}, but none had a native target inside the configured radius.")
            recommendations.append("Repeat validation while standing near at least one selected monster.")
        else:
            add("eva_candidate_input", "pass", f"{self._casts_with_candidates}/{self._casts} casts had candidates; total candidates={self._cast_candidate_total}.")

        if self._ocr_positive_delta > 0 and self._native_kills == 0:
            if self._casts_with_candidates <= 0:
                cause = (
                    "OCR increased, but the EVA casts had no native candidates "
                    "inside the configured radius."
                )
                recommendation = (
                    "Check actor distances, EVA radius scaling, and whether the "
                    "selected monsters are present in the native observation."
                )
            elif (
                self._candidate_traces > 0
                and self._candidate_always_present == self._candidate_traces
            ):
                cause = (
                    "OCR increased, but every native candidate remained visible "
                    "through all post-cast polls."
                )
                recommendation = (
                    "The recovered monster HP/presence fields or cached actor-slot "
                    "state are likely not reflecting death."
                )
            elif self._candidate_single_absence > 0:
                cause = (
                    "OCR increased, but at least one candidate disappeared for only "
                    "one successful poll; two consecutive absence reads are required."
                )
                recommendation = (
                    "Review cast polling timing and slot reuse before changing the "
                    "confirmation threshold."
                )
            elif self._kill_poll_failed_reads > 0:
                cause = (
                    "OCR increased while native confirmation suffered failed actor "
                    "reads during the post-cast window."
                )
                recommendation = (
                    "Inspect pointer/cache availability during EVA confirmation."
                )
            else:
                cause = (
                    f"OCR increased by {self._ocr_positive_delta}, but native "
                    "confirmed kills stayed at 0."
                )
                recommendation = (
                    "Inspect candidate HP and presence transitions in events.jsonl."
                )
            add("kill_channel_agreement", "fail", cause)
            recommendations.append(recommendation)
        elif self._native_kills > 0 and self._ocr_positive_delta == 0:
            add("kill_channel_agreement", "warn", f"Native confirmed {self._native_kills} kills, but OCR showed no accepted increase.")
            recommendations.append("The OCR channel missed kills; inspect included frames and dynamic panel anchoring.")
        elif self._native_kills > 0 and self._ocr_positive_delta > 0:
            status = "pass" if self._native_kills == self._ocr_positive_delta else "warn"
            add("kill_channel_agreement", status, f"Native kills={self._native_kills}; OCR delta={self._ocr_positive_delta}.")
            if status == "warn":
                recommendations.append("Review timing around casts; the two kill channels disagreed on the count.")
        else:
            add("kill_channel_agreement", "inconclusive", "Neither channel observed a kill during this run.")
            recommendations.append("Repeat the validation until at least one visible monster is killed.")

        if self._errors:
            add("debug_artifact_integrity", "warn", f"Recorder errors: {self._errors}")
        else:
            add("debug_artifact_integrity", "pass", "The diagnostic package was recorded without internal errors.")

        statuses = {str(item["status"]) for item in checks}
        if "fail" in statuses:
            verdict = "fail"
        elif "inconclusive" in statuses:
            verdict = "inconclusive"
        elif "warn" in statuses:
            verdict = "warn"
        else:
            verdict = "pass"
        return checks, verdict, recommendations

    def finish(
        self,
        *,
        session_reason: str,
        session_classification: str,
        preflight: Mapping[str, object],
        extra: Mapping[str, object] | None = None,
    ) -> ValidationArtifacts:
        if self._finished:
            return ValidationArtifacts(self.session_directory, self.events_path, self.summary_path, self.archive_path)
        self._finished = True
        self._events_handle.flush()
        self._events_handle.close()
        checks, verdict, recommendations = self._build_checks()
        summary = {
            "version": 1,
            "verdict": verdict,
            "session_reason": str(session_reason),
            "session_classification": str(session_classification),
            "preflight": self._json_safe(dict(preflight)),
            "counts": {
                "events": self._event_count,
                "steps": self._step_count,
                "actions": dict(self._action_counts),
                "native_kills": self._native_kills,
                "ocr_baseline": self._ocr_baseline,
                "ocr_latest": self._ocr_latest,
                "ocr_positive_delta": self._ocr_positive_delta,
                "ocr_outcomes": dict(self._ocr_outcomes),
                "eva_casts": self._casts,
                "casts_with_candidates": self._casts_with_candidates,
                "cast_candidate_total": self._cast_candidate_total,
                "confirmed_actor_addresses": sorted(self._confirmed_base_addresses),
                "kill_poll_successful_reads": self._kill_poll_successful_reads,
                "kill_poll_failed_reads": self._kill_poll_failed_reads,
                "candidate_traces": self._candidate_traces,
                "candidate_always_present": self._candidate_always_present,
                "candidate_single_absence": self._candidate_single_absence,
                "candidate_confirmed_traces": self._candidate_confirmed_traces,
                "actor_samples": self._actor_samples,
                "visible_actor_minimum": self._minimum_visible_actors,
                "visible_actor_maximum": self._maximum_visible_actors,
                "observed_species": {str(key): value for key, value in sorted(self._observed_species.items())},
                "movement_steps": self._movement_steps,
                "movement_steps_with_displacement": self._movement_steps_with_displacement,
                "contacts": self._contacts,
                "hp_decreases": self._hp_decreases,
                "hp_zero_transitions": self._hp_zero_transitions,
                "candidate_disappearances": self._candidate_disappearances,
                "pointer_modes": dict(self._pointer_modes),
                "pointer_generations": {str(key): value for key, value in sorted(self._pointer_generations.items())},
                "observation_sizes": {str(key): value for key, value in sorted(self._observation_sizes.items())},
                "nonfinite_observations": self._nonfinite_observations,
                "screenshots": self._screenshots,
            },
            "checks": checks,
            "recommendations": recommendations,
            "recorder_errors": list(self._errors),
            "extra": self._json_safe(dict(extra or {})),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.readme_path.write_text(
            "FlyFF training-data validation package\n"
            "======================================\n\n"
            f"Automatic verdict: {verdict.upper()}\n\n"
            "Upload this entire ZIP to ChatGPT. The important files are:\n"
            "- summary.json: automatic checks and high-level counts\n"
            "- events.jsonl: full per-step native/OCR/action/reward/observation evidence\n"
            "- frames/: selected game screenshots for OCR and cast diagnosis\n\n"
            "Do not edit or extract individual files before uploading; the complete package lets the data channels be correlated.\n",
            encoding="utf-8",
        )
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(self.session_directory.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(self.session_directory))
        return ValidationArtifacts(self.session_directory, self.events_path, self.summary_path, self.archive_path)

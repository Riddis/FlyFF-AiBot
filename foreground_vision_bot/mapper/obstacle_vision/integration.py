from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from .ActiveLearning import decide_active_learning
from .DatasetRecorder import ObstacleDatasetRecorder, ObstacleSampleLabel
from .FloorAppearance import OnlineFloorAppearanceModel
from .Prompt import request_label
from .Transforms import build_review_image

_PATCH_FLAG = "_obstacle_vision_v21_dropin_installed"


def install_obstacle_vision(mapper_cls: type) -> None:
    """Attach v2.1 collection to a v1.9.7 AdaptiveMapper without replacing it."""
    if getattr(mapper_cls, _PATCH_FLAG, False):
        return
    original_init = mapper_cls.__init__
    original_run = mapper_cls._run
    original_forward = mapper_cls._execute_forward

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        app_root = Path(
            os.environ.get("FLYFF_OBSTACLE_VISION_ROOT", str(Path(__file__).resolve().parents[2]))
        ).resolve()
        config = self.config
        crop_fraction = float(getattr(config, "obstacle_dataset_crop_fraction", 0.72))
        image_size = int(getattr(config, "obstacle_dataset_image_size", 224))
        run_id = _run_id(self)
        map_name = str(getattr(getattr(self, "map_profile", None), "name", "unknown"))
        self._obstacle_v21_floor = OnlineFloorAppearanceModel(
            app_root / "models" / "mapping" / "floor_appearance.npz",
            crop_fraction=crop_fraction,
            image_size=image_size,
        )
        self._obstacle_v21_recorder = ObstacleDatasetRecorder(
            app_root / "datasets" / "obstacle_vision" / "raw",
            run_id=run_id,
            map_name=map_name,
            enabled=bool(getattr(config, "obstacle_dataset_enabled", True)),
            crop_fraction=crop_fraction,
            image_size=image_size,
            jpeg_quality=int(getattr(config, "obstacle_dataset_jpeg_quality", 90)),
            deduplicate_clear=bool(getattr(config, "obstacle_clear_deduplicate", True)),
            minimum_clear_hash_distance=int(
                getattr(config, "obstacle_clear_hash_distance", 7)
            ),
            keep_clear_every=int(getattr(config, "obstacle_keep_clear_every", 40)),
        )
        self._obstacle_v21_prompt_enabled = bool(
            getattr(config, "obstacle_active_learning_prompt_enabled", True)
        )
        self._obstacle_v21_prompt_on_disagreement = bool(
            getattr(config, "obstacle_prompt_on_disagreement", True)
        )
        self._obstacle_v21_step = 0

    def patched_run(self: Any) -> Any:
        _status(self, "Automatic floor/not-floor model: " + self._obstacle_v21_floor.status)
        _status(
            self,
            "Obstacle collection is active. Clear travel teaches floor automatically; "
            "only unresolved or contradictory views request a label.",
        )
        try:
            return original_run(self)
        finally:
            try:
                self._obstacle_v21_floor.save()
            finally:
                self._obstacle_v21_recorder.close()

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        step = _step_number(self, args, kwargs)
        heading = _heading(self)
        captured: list[Any] = []
        original_wait = self._wait_for_frame_sample

        def capturing_wait(*wait_args: Any, **wait_kwargs: Any) -> Any:
            sample = original_wait(*wait_args, **wait_kwargs)
            if getattr(sample, "frame", None) is not None:
                captured.append(sample)
            return sample

        self._wait_for_frame_sample = capturing_wait
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            self._wait_for_frame_sample = original_wait

        if not captured:
            return result
        before = captured[0]
        after = getattr(result, "frame_sample", None) or captured[-1]
        if getattr(after, "frame", None) is None:
            return result

        label, confidence, reason = _infer_label(result)
        camera_obscured = _truthy_attr(result, "camera_obscured")
        teleport_likely = "teleport" in reason.lower()
        floor_prediction = self._obstacle_v21_floor.predict(
            before.frame,
            heading_deg=heading,
        )
        decision = decide_active_learning(
            motion_label=label.value,
            motion_confidence=confidence,
            camera_obscured=camera_obscured,
            teleport_likely=teleport_likely,
            floor_prediction=floor_prediction,
            prompt_on_disagreement=self._obstacle_v21_prompt_on_disagreement,
        )

        label_source = "motion"
        review_requested = bool(decision.prompt)
        if decision.prompt and self._obstacle_v21_prompt_enabled:
            review_image = build_review_image(
                before.frame,
                heading_deg=heading,
                crop_fraction=self._obstacle_v21_floor.crop_fraction,
                image_size=448,
                title=f"Step {step}: classify highlighted path",
            )
            response = request_label(
                review_image,
                motion_label=label.value,
                suggested_label=decision.suggested_label.value,
                reason=decision.reason,
                floor_risk=floor_prediction.obstacle_risk,
            )
            if response in {"clear", "blocked", "ignore"}:
                label = ObstacleSampleLabel(response)
                confidence = 1.0 if response != "ignore" else confidence
                label_source = "human"
                reason = f"human_floor_review:{response}; {reason}"
                _status(self, f"Obstacle label accepted: step {step} -> {response}.")
            else:
                label = ObstacleSampleLabel.IGNORE
                reason = f"queued_for_review; {reason}"

        if label is ObstacleSampleLabel.CLEAR and (
            label_source == "human" or confidence >= 0.65
        ):
            try:
                self._obstacle_v21_floor.observe_clear(
                    before.frame,
                    heading_deg=heading,
                )
            except Exception as error:  # collection must never stop mapping
                _status(self, f"Floor learning failed open: {type(error).__name__}: {error}")

        try:
            sample = self._obstacle_v21_recorder.record(
                step=step,
                before_frame=before.frame,
                after_frame=after.frame,
                heading_deg=heading,
                label=label,
                confidence=confidence,
                reason=reason,
                label_source=label_source,
                review_requested=review_requested,
                floor_risk=floor_prediction.obstacle_risk,
                metadata={
                    "camera_obscured": camera_obscured,
                    "floor_model_status": floor_prediction.status,
                },
            )
            retained = sample is not None and sample.retained
            risk_text = (
                "unknown"
                if floor_prediction.obstacle_risk is None
                else f"{floor_prediction.obstacle_risk:.2f}"
            )
            _status(
                self,
                f"obstacle_sample={label.value} retained={retained} "
                f"floor_risk={risk_text}; {self._obstacle_v21_floor.status}",
            )
        except Exception as error:  # collection must never stop mapping
            _status(self, f"Obstacle recording failed open: {type(error).__name__}: {error}")
        return result

    mapper_cls.__init__ = patched_init
    mapper_cls._run = patched_run
    mapper_cls._execute_forward = patched_forward
    mapper_cls.VERSION = "2.1.0-automated-floor-active-learning-dropin"
    setattr(mapper_cls, _PATCH_FLAG, True)


def _run_id(mapper: Any) -> str:
    output_dir = getattr(mapper, "output_dir", None)
    if output_dir is not None:
        name = Path(output_dir).name
        if name:
            return name
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _step_number(mapper: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    if args:
        try:
            return int(args[0])
        except (TypeError, ValueError):
            pass
    if "step" in kwargs:
        try:
            return int(kwargs["step"])
        except (TypeError, ValueError):
            pass
    mapper._obstacle_v21_step += 1
    return int(mapper._obstacle_v21_step)


def _heading(mapper: Any) -> float:
    try:
        return float(mapper.grid.continuous_pose.heading_deg)
    except (AttributeError, TypeError, ValueError):
        try:
            return float(mapper.grid.pose.heading_deg)
        except (AttributeError, TypeError, ValueError):
            return 0.0


def _infer_label(result: Any) -> tuple[ObstacleSampleLabel, float, str]:
    motion = _text_attr(result, "motion")
    reason = _text_attr(result, "stop_reason") or _text_attr(result, "recovery_reason")
    combined = f"{motion} {reason}".lower()
    if any(word in combined for word in ("blocked", "contact", "obstacle")):
        return ObstacleSampleLabel.BLOCKED, 0.82, reason or motion or "blocked"
    if any(word in combined for word in ("moved", "clear", "translation")):
        return ObstacleSampleLabel.CLEAR, 0.86, reason or motion or "moved"
    distance = getattr(result, "distance_cells", None)
    try:
        numeric_distance = float(distance)
    except (TypeError, ValueError):
        numeric_distance = 0.0
    if numeric_distance >= 0.5:
        return ObstacleSampleLabel.CLEAR, 0.75, "accepted_forward_distance"
    return ObstacleSampleLabel.IGNORE, 0.0, reason or motion or "uncertain_forward_outcome"


def _text_attr(value: Any, name: str) -> str:
    item = getattr(value, name, None)
    if item is None:
        return ""
    enum_value = getattr(item, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    enum_name = getattr(item, "name", None)
    if isinstance(enum_name, str):
        return enum_name
    return str(item)


def _truthy_attr(value: Any, name: str) -> bool:
    return bool(getattr(value, name, False))


def _status(mapper: Any, message: str) -> None:
    callback = getattr(mapper, "status_callback", None)
    if callable(callback):
        try:
            callback(str(message))
        except Exception:
            pass

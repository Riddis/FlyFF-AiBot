"""Recording-purpose provenance (docs/PROJECT_GOALS.md section 6).

Every recording is classified by PURPOSE, not merely by who controlled
it. This module defines that classification once, shared by both the
automatic operational-feedback path (RuntimeController) and the
explicit controlled-experiment path (Gui.py's compact Recording
section) -- neither duplicates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RecordingPurpose = Literal["OPERATIONAL_FEEDBACK", "CONTROLLED_EXPERIMENT"]
ControllerType = Literal["HUMAN_CONTROLLED", "BOT_POLICY_CONTROLLED", "SCRIPTED_CONTROLLED"]
DataUseRole = Literal["FITTING_ELIGIBLE", "VALIDATION_HOLDOUT", "DIAGNOSTIC_ONLY"]


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    """Recorded verbatim into manifest.json's ``experiment_provenance``
    block. Never inferred after the fact from a recording's content --
    decided at recording start, per docs/PROJECT_GOALS.md section 6."""

    purpose: RecordingPurpose = "OPERATIONAL_FEEDBACK"
    controller_type: ControllerType = "BOT_POLICY_CONTROLLED"
    protocol_id: str | None = None
    hypothesis: str | None = None
    data_use_role: DataUseRole = "FITTING_ELIGIBLE"

    def __post_init__(self) -> None:
        if self.purpose not in ("OPERATIONAL_FEEDBACK", "CONTROLLED_EXPERIMENT"):
            raise ValueError(f"Unknown recording purpose: {self.purpose!r}")
        if self.controller_type not in (
            "HUMAN_CONTROLLED",
            "BOT_POLICY_CONTROLLED",
            "SCRIPTED_CONTROLLED",
        ):
            raise ValueError(f"Unknown controller type: {self.controller_type!r}")
        if self.data_use_role not in (
            "FITTING_ELIGIBLE",
            "VALIDATION_HOLDOUT",
            "DIAGNOSTIC_ONLY",
        ):
            raise ValueError(f"Unknown data-use role: {self.data_use_role!r}")
        if self.purpose == "CONTROLLED_EXPERIMENT" and not self.protocol_id:
            raise ValueError(
                "CONTROLLED_EXPERIMENT recordings must carry a protocol_id -- "
                "a deliberately designed session answers a specific, "
                "predeclared question, per docs/PROJECT_GOALS.md section 6"
            )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "purpose": self.purpose,
            "controller_type": self.controller_type,
            "protocol_id": self.protocol_id,
            "hypothesis": self.hypothesis,
            "data_use_role": self.data_use_role,
        }


OPERATIONAL_FEEDBACK_DEFAULT = ExperimentProvenance(
    purpose="OPERATIONAL_FEEDBACK",
    controller_type="BOT_POLICY_CONTROLLED",
    data_use_role="FITTING_ELIGIBLE",
)

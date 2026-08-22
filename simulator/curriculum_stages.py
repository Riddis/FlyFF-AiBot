"""Canonical project-level curriculum stage names, mapped onto the existing
internal stage identifiers ("early"/"intermediate"/"advanced") used
throughout synthetic.py, curriculum manifests, and generated map/world
assets.

This is the ONE place that translates between the two vocabularies. It does
not rename anything: existing curricula, manifests, and generation code keep
using "early"/"intermediate"/"advanced" as their stage identifier, since a
mass rename would touch generated assets, manifests, tests, and reports for
no behavioral benefit. New pipeline code, configs, checkpoint names, and
reports should use the canonical names below and look up the internal stage
through CANONICAL_STAGES rather than inventing another spelling.

Canonical stages, in training order:

  Basic        Assisted training regime. The Phase 1 recovery controller
               (simulator.recovery_controller.RecoveryController) is allowed
               to intervene during training -- training wheels, not part of
               the learned policy. Reuses Beginner's geometry (internal
               stage "early"), optionally a simpler subset of it; Basic is
               distinguished from Beginner by recovery being enabled during
               training/eval, not by a different map generator or a
               different internal stage identifier. Graduates on stable
               assisted competence (useful target/event/navigation learning,
               no collapse, recovery reliance trending down) -- NOT on raw,
               recovery-off performance. Raw Beginner-level competence is
               Beginner's graduation bar, not Basic's.

  Beginner     Recovery removed. Internal stage "early". Independent
               navigation on the existing early-stage synthetic templates
               (open_field, irregular_plain, broad_lobes, wide_neck,
               split_field, open_center). Graduates on raw-policy
               competence: normal navigation without walking into
               obstacles, not merely reduced stagnation.

  Intermediate Internal stage "intermediate". Unchanged in scope: real
               obstacle routing, route choice, recoverable bad approaches.

  Advanced     Internal stage "advanced". Unchanged in scope: hardest
               combined geometry/density/respawn conditions.

The escapability gate in synthetic.py (_STAGE_ESCAPE_TICKS) is a map-sanity
constraint keyed by the internal stage identifier -- it rejects
pathologically unforgiving generated geometry, it is not a description of
what a graduated policy is expected to do. See that module's own comment for
the full derivation.
"""

from __future__ import annotations

from typing import Final

# canonical name -> internal stage identifier used by synthetic.py /
# curriculum manifests / generated map & world assets.
CANONICAL_STAGES: Final[dict[str, str]] = {
    "basic": "early",
    "beginner": "early",
    "intermediate": "intermediate",
    "advanced": "advanced",
}

# Which canonical stages use recovery assistance during training. Basic is
# the only one -- this is the actual distinction between Basic and
# Beginner, not the internal stage identifier (both are "early").
RECOVERY_ASSISTED_STAGES: Final[frozenset[str]] = frozenset({"basic"})


def internal_stage(canonical_name: str) -> str:
    """Resolve a canonical stage name ("basic", "beginner", "intermediate",
    "advanced") to the internal stage identifier used by synthetic.py and
    curriculum manifests. Raises KeyError with the valid options listed if
    the name isn't recognized."""

    key = canonical_name.strip().lower()
    if key not in CANONICAL_STAGES:
        raise KeyError(
            f"Unknown canonical stage {canonical_name!r}; expected one of "
            f"{sorted(CANONICAL_STAGES)}"
        )
    return CANONICAL_STAGES[key]


def uses_recovery_assistance(canonical_name: str) -> bool:
    """Whether training at this canonical stage allows the Phase 1 recovery
    controller to intervene. Only "basic" does; every later stage graduates
    on raw-policy performance with recovery disabled."""

    key = canonical_name.strip().lower()
    if key not in CANONICAL_STAGES:
        raise KeyError(
            f"Unknown canonical stage {canonical_name!r}; expected one of "
            f"{sorted(CANONICAL_STAGES)}"
        )
    return key in RECOVERY_ASSISTED_STAGES

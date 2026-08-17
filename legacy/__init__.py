"""Historical/version-specific archive-manifest compatibility.

Nothing outside this package (and ``simulator.schema``, its sole canonical
caller) should import from here directly -- see R7b in
``CANONICAL_OWNERS.toml``. Ordinary product code uses ``simulator.schema``'s
public surface, which dispatches into this package only for the specific,
enumerated absence-driven compatibility decisions documented in
``docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`` sections C and F.
"""

from __future__ import annotations

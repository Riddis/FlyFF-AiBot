# Canonical Dev App — First Live Acceptance Campaign

**STATUS: PAUSED — two real product regressions found in run 1; fixes
applied offline; human live campaign resumes at the user's discretion.**

This is a distinct campaign from G5 (`G5_REAL_CLIENT_VALIDATION.md`,
position/pointer-recovery specific). This one validates `apps/dev_app.py`
itself — launch, attach, GUI usability, and normal shutdown — as a
whole, end to end, for the first time since consolidation.

**No agent may execute any part of this procedure.** Agents prepare it
and analyze returned evidence; the user runs it. See
`docs/agent/PROJECT_RULES.md`.

## Run 1 — 2026-08-20

- Test date: 2026-08-20.
- Tested revision: `dceac0e158a7e847d6e6cc6137023b67789163a9`.
- Branch: `refactor/consolidation-phase1`.
- Evidence preserved locally under `live_validation/20260820_171803/`
  (gitignored — screenshots, `H2_H19_gui_crash.log`,
  `human_findings.txt`; referenced here by path, not committed as
  binary evidence).

### USER-RUN LIVE OBSERVATION (evidence distinction preserved verbatim)

1. `python -m apps.dev_app` launched successfully.
2. The canonical application successfully attached to the real FlyFF
   client.
3. Bot vision/OCR appeared operational.
4. **FAIL — GUI layout.** The sidebar/Actions controls were severely
   vertically stretched; several buttons were not practically
   readable/clickable. Session Statistics rendered; content below it
   did not size correctly.
5. **FAIL — shutdown.** Closing the application normally raised:
   ```
   File "apps/dev_app.py", line 28, in main
       gui.loop(bot)
   File "Gui.py", line 121, in loop
       self.__refresh_runtime(values)
   File "Gui.py", line 380, in __refresh_runtime
       and values.get("-SHOW_FRAMES-", False)
   AttributeError: 'NoneType' object has no attribute 'get'
   ```
   Crash log saved to `apps/gui_crash.log` (local, gitignored).

**Only H1 (cold start)/attach/vision are confirmed working. Nothing
past H2 was exercised — the human live campaign paused here per the
user's own report; untested items are not claimed to pass.**

### Root-cause investigation (offline, source-only — no live execution by the agent)

**GUI layout.** Traced PySimpleGUI's construction-time layout logic
(`PackFormIntoFrame`/`_add_expansion` in the installed
`PySimpleGUI==4.60.5.1`) through three nesting levels (the scrollable
`-MAIN_COLUMN-` Column → each inner `sg.Frame` → each button row) and
found the row-expansion logic itself correctly gates on `expand_x`
alone (does not propagate to vertical row expansion) — no code-level
defect found there. `git log --follow` confirms `Gui.py` has not
changed in this area since before the Phase-7 collapse (`bfc5c6d`);
Phase 10's only touch (`0f9b7b2`) added a new `dev_tools` row to the
pre-existing `controls` Column and did not touch any `expand_x`/
`expand_y`/`size`/`scrollable` parameter. **This is not attributable to
any Phase-10/13/14 diff** — the layout code is unchanged; this is the
first time it has ever been exercised on a real, physically-scaled
display.

A concrete, confirmed gap was found instead: nothing in this codebase
ever declares Windows per-monitor DPI awareness before creating the Tk
window (`grep` for `SetProcessDpiAwareness`/`SetProcessDPIAware`
returns nothing prior to this fix). Without that declaration, Windows
applies its own bitmap compatibility scaling to the whole
(DPI-unaware) process while Tk's font-driven widget heights still
scale to the display's real DPI internally — the two `Gui.py` Column
`size=` constants (`(1320, 990)` for the window, `(335, 820)` for the
scrollable sidebar canvas) are raw pixel values that only match
intended proportions at the DPI they were tuned against. This mismatch
is the most concrete, evidence-backed explanation consistent with
every observed symptom (disproportionate button height, right-edge
clipping, a scrollbar appearing for content that should fit) and with
the fact that it was never caught by any headless/offline test (no
real display DPI exists in that context) or by any prior human run
(this was the first one).

**Fix applied**: `apps/dev_app.py` now declares Per-Monitor-v2 DPI
awareness (`_declare_windows_dpi_awareness()`, Windows-only,
best-effort, called as the first statement in `main()`, before
`gui.init()` creates the real window). This is the standard, minimal,
well-documented remediation for this class of symptom and does not
alter any layout code. **This is a hypothesis-driven, evidence-backed
fix, not an empirically-confirmed one — the agent cannot launch the
live GUI to verify pixel-perfect rendering. Final visual acceptance
remains USER-RUN** (see the retest procedure below).

**Shutdown crash.** Confirmed directly from the traceback and
`Gui.py`'s source: PySimpleGUI's `Window.read()` returns `values=None`
alongside `sg.WIN_CLOSED` (the window and its elements are already
gone). `Gui.loop()` called `self.__refresh_runtime(values)`
unconditionally, before checking for `WIN_CLOSED` — crashing on
`values.get(...)` instead of ever reaching the normal
`__shutdown(bot)` cleanup path. The crash propagated out to
`apps/dev_app.py`'s top-level `except`/`finally`, which does contain a
fallback that calls `gui.controller.shutdown(...)` if not already
finalized — so no resource was actually leaked in this run — but the
intended single, clean shutdown path never executed, and the user saw
an unexpected error dialog on a normal close.

**Fix applied**: `Gui.loop()` now checks `values is None or event ==
sg.WIN_CLOSED` immediately after `window.read()`, before any element
read/update, and routes straight to the existing `__shutdown(bot)`
path in that case — a narrow reordering, not new shutdown logic. See
`tests/test_gui_event_loop_lifecycle.py` (confirmed failing before this
fix, per this document's own re-verification, and passing after).

### Fix commit

See `COMMAND_LOG.tsv`'s `G1-LIVE-FIX` row for the exact commit hash.

### Criterion-by-criterion result

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Cold start / attach / vision | PASS | User-observed, run 1 |
| 2 | GUI layout usable at supported window sizes | **FAIL → fix applied, offline-unverifiable** | Screenshots, `human_findings.txt` |
| 3 | Normal close produces no error | **FAIL → fixed, regression-tested offline** | `H2_H19_gui_crash.log`; `tests/test_gui_event_loop_lifecycle.py` |
| 4-7 | Resize/re-attach/relaunch stability | NOT YET RUN | Campaign paused before reaching these steps |

### Confidence

- Shutdown fix: HIGH — the exact crash mechanism was read directly
  from the traceback and source, the fix is a narrow reordering, and
  the regression test fails pre-fix / passes post-fix.
- Layout fix: MEDIUM — a well-documented, common, and evidence-
  consistent root cause (absence of DPI-awareness declaration) with a
  standard, minimal, safe remediation, but not empirically confirmed
  against the user's real display. Genuinely possible a second,
  independent contributing factor remains and requires a follow-up
  finding after retest.

### Follow-up

User retest required (see `docs/operations/DEVELOPMENT_WORKFLOWS.md`
retest procedure, or the exact steps handed back at the end of this
fix). If the layout is still wrong after this fix, the next
investigation step is to gather the user's actual Windows display
scaling percentage and a fresh screenshot, and reconsider whether the
`controls` Column's fixed pixel `size=(335, 820)` needs to become
content-derived rather than a magic constant.

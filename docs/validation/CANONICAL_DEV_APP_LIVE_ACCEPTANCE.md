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

### FALSIFIED HYPOTHESIS: DPI / Windows display scaling

An initial investigation traced PySimpleGUI's construction-time layout
logic (`PackFormIntoFrame`/`_add_expansion`) through three nesting
levels and found no code-level row-expansion defect, then proposed
that nothing in the codebase declared Windows per-monitor DPI
awareness as the root cause, and shipped
`apps/dev_app.py._declare_windows_dpi_awareness()` as a fix — **without
ever constructing the real window and measuring its actual rendered
geometry first.** The user retested: no visible change, and reported
that the same installed packages had rendered this exact sidebar
correctly before the migration — directly falsifying the hypothesis. A
follow-on PySimpleGUI-version-drift theory (installed `4.60.5.1` vs. a
`4.60.3` comment in `requirements.txt`, `4.60.3` no longer available on
PyPI) was also chased and also falsified by the same "same packages,
worked before" fact. **The DPI-awareness change was reverted.** Neither
Windows display scaling, DPI awareness, the `ttk` "vista" theme (just
Tk's internal identifier for the native-Windows renderer — unrelated
to the Windows Vista OS), nor a package-version difference is
implicated. See `MISTAKES.md`'s "[2026-08-20]" entries for the full
account, including the process lesson (measure real rendered geometry
before proposing a root cause).

### SOURCE/LAYOUT ANALYSIS: the actual, measured root cause

Constructing the real Tk window directly (`Gui().init()` — no bot, no
FlyFF attach, no native reader, pure local geometry) and reading its
actual `winfo_reqwidth()`/`winfo_width()` values found the mechanism
immediately: Phase 10 added a four-column, 313+-row artifact `sg.Table`
(`col_widths=[10, 30, 18, 24]`, `expand_x=True`,
`key="-DEVTOOLS-ARTIFACTS-TABLE-"`) as a direct child of `-MAIN_COLUMN-`
— the sidebar's fixed-width (335px), vertically-scroll-only,
`scrollable=True` Column. Measured: the Table's real requested width
was ~740px; the whole Column's inner frame (`TKFrame.winfo_reqwidth()`)
was 779px against a 335px canvas (`canvas.winfo_reqwidth()`); the
`-VALIDATE_DATA-` button rendered at 755px actual width. Because
`vertical_scroll_only=True` disables horizontal scrolling, that ~450px
of excess width is simply clipped by the canvas, not reachable by
scrolling. Every sibling `expand_x=True` control (all of Actions,
Redetect UI Panels, Show Log, Launch, Cancel) filled to that oversized
779px inner width and had its centered label pushed partially or fully
outside the visible 335px — exactly matching every observed symptom
(apparently blank buttons, text visible only at the far right, one
button of a paired row missing entirely). `git log --follow` confirms
`Gui.py`'s pre-Phase-10 layout code was otherwise unchanged; Phase 10's
own commit (`0f9b7b2`) is the one that added this Table, and its own
GUI tests deliberately never called `Gui.init()`, so they could not
have caught a real-rendered-geometry regression like this one.

**Fix applied**: the artifact table was removed from `-MAIN_COLUMN-`
and moved to its own separate, resizable window
(`Gui.__show_artifact_window`, opened on demand via a compact "View
Artifact Inventory" button plus a cheap row-count summary in the
sidebar, serviced non-blockingly alongside the pre-existing
`_log_window` pattern). Measured post-fix: inner frame width dropped
from 779px to 398px (canvas still 335px — a modest ~63px residual from
pre-existing, unrelated `size=(N chars, ...)` Text elements), and the
widest sidebar button dropped from 755px to 374px actual width.
`tests/test_gui_sidebar_geometry.py` asserts on these real measured
widths directly (confirmed failing against the pre-fix layout, passing
post-fix) — a genuine improvement over the prior fix, which had no
test against real geometry at all. **Final visual acceptance on
the user's actual display remains USER-RUN** (see the retest procedure
below).

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

### Fix commits

Two forward commits: an initial (later-reverted) DPI-awareness fix,
and the corrected structural fix (artifact table moved out of
`-MAIN_COLUMN-`). See `COMMAND_LOG.tsv`'s `G1-LIVE-FIX` and
`G1-LIVE-FIX-CORRECTION` rows for the exact commit hashes.

### Criterion-by-criterion result

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Cold start / attach / vision | PASS | User-observed, run 1 |
| 2 | GUI layout usable at supported window sizes | **FAIL → structurally fixed and measured offline; visual confirmation still USER-RUN** | Screenshots, `human_findings.txt`; `tests/test_gui_sidebar_geometry.py` |
| 3 | Normal close produces no error | **FAIL → fixed, regression-tested offline** | `H2_H19_gui_crash.log`; `tests/test_gui_event_loop_lifecycle.py` |
| 4-7 | Resize/re-attach/relaunch stability | NOT YET RUN | Campaign paused before reaching these steps |

### Confidence

- Shutdown fix: HIGH — the exact crash mechanism was read directly
  from the traceback and source, the fix is a narrow reordering, and
  the regression test fails pre-fix / passes post-fix.
- Layout fix: HIGH on the mechanism, MEDIUM-HIGH overall — the root
  cause is directly measured (real `winfo_reqwidth()`/`winfo_width()`
  values from the actual rendered window, not inferred), the fix
  removes the exact widget that caused the measured overflow, and a
  geometry regression test confirms the pre/post-fix numbers.
  Downgraded from HIGH only because the user's real display still has
  not been checked after this specific fix — an initial (wrong)
  hypothesis was already shipped once without that check, so this
  document does not repeat that overclaim.

### Follow-up

User retest required (see the retest procedure below). If the sidebar
is still visibly wrong after this fix, the residual ~63px inner-frame
overflow (from pre-existing `size=(N chars, ...)` Text elements,
unrelated to this regression) is the next place to look — but the
~450px overflow that caused the reported symptom is now removed and
measured gone.

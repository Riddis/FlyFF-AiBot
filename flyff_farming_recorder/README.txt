FLYFF FARMING SESSION RECORDER 1.11
==================================

This is a read-only recorder. It does not control FlyFF or modify game memory.

HOW TO RECORD
-------------
1. Log in and enter the Tower farming map.
2. Stand at the map spawn and make sure your HP is full.
3. Open the recorder.
4. Select your FlyFF client.
5. Enter your current full HP.
6. Select your keyboard layout and EVA hotkey.
7. Click Attach Client and wait until it finishes.
8. Click Start Logging.
9. Farm normally.
10. Click End Logging when finished.
11. Send the generated SEND_TO_RIDDIMS_*.zip file to Riddims.

The ZIP is saved in:
Documents\FlyffFarmingRecorder

If you leave or teleport out of the farming map before pressing End Logging, the
recorder automatically ends and packages the session.

EVA HOTKEYS
-----------
The EVA list contains F1-F12 and 1-9. Number choices refer to the physical
number-row keys. On AZERTY, choose 2 for the é key, 1 for &, and so on.

AUTOMATIC CLASSIFICATION
------------------------
There is no role or movement-method selector. Do not edit recorder_config.json to
classify a session.

The recorder compares observed displacement with the recorded W/Z, Q/A, D,
Space, and EVA key state. At the end it automatically classifies movement as:

- keyboard_wasd
- click_to_move
- mixed
- unknown

A direct keyboard session may become behavior-cloning data. Click-to-move and
mixed movement never become direct low-level movement labels. Their real EVA
key events can still be retained, and a presence-validated session can still be
used as world-model data.

PRESENCE FIELD RECOVERY
-----------------------
The instantiated/presence field is never trusted solely because its offset looks
familiar. The recorder first tries an exact-client-build recovery profile, then
revalidates it against the current process. If the game executable changed or
validation fails, full dynamic recovery runs.

During the recording, longitudinal live-versus-dormant evidence can promote a
strong candidate and enable hot/cold polling. Same-slot reappearance is only
diagnostic because actor addresses are reusable pool slots. A validated offset
is saved with the exact Neuz.exe fingerprint so a normal restart can recover it
quickly; a changed client build cannot silently reuse it.

The final recorder.log and manifest.json state whether the archive is eligible
for authoritative world-model fitting, direct demonstrations, and/or EVA-only
export. Do not use population, density, disappearance, or respawn statistics as
authoritative unless world_model_eligible is true.

NORMAL USE
----------
Farm exactly as you normally would. Keyboard users can hold W/Z and add Q/A or D
while turning; the combined actions are recorded correctly. For useful
world-model evidence, cover several normal farming regions rather than remaining
in one tiny loaded area for the whole session.

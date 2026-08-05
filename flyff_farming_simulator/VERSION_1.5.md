# Simulator 1.5 — Synthetic Open-Farm Curriculum

- Adds 12 generated large, open farming layouts across early, intermediate, and advanced stages.
- Excludes mazes, dungeon room chains, long corridors, and precision-navigation layouts.
- Adds low, typical, high, uneven, and shifting monster-density profiles.
- Adds fast, typical, variable, bursty, and slow respawn profiles.
- Adds loose clustered monster spawn reservoirs and redistribution-heavy variants.
- Starts every generated episode at one designated map spawn.
- Adds staged generic-base training and cross-layout evaluation commands.
- Uses the real recorded movement and cast timings without copying Tower geometry.
- Fixes future recorded-world fitting so only the first focused farming position from each session is used as a spawn candidate.
- Keeps the production five-action and 923-value observation contract.

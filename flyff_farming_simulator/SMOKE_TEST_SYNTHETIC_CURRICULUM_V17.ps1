[CmdletBinding()]
param(
    [ValidateSet("early", "intermediate", "advanced", "all")]
    [string]$Stage = "early",
    [double]$EpisodeSeconds = 10.0,
    [int]$MaxActions = 100,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
if (Test-Path "..\.venv\Scripts\python.exe") {
    $python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
}

& $python -u -c @"
import json
from simulator.reward_model import REWARD_CONTRACT_ID
from simulator.synthetic import iter_variant_environments

results = []
for index, (entry, env) in enumerate(iter_variant_environments(
    r'synthetic_curriculum\curriculum.json',
    stage='$Stage',
    seed=$Seed,
    episode_steps=$MaxActions,
    episode_seconds=$EpisodeSeconds,
)):
    observation, info = env.reset(seed=$Seed + index)
    reward_total = 0.0
    last = info
    for _ in range($MaxActions):
        action = int(env.rng.integers(0, 5))
        observation, reward, terminated, truncated, last = env.step(action)
        reward_total += float(reward)
        if terminated or truncated:
            break
    component_total = sum(last.get('reward_component_totals', {}).values())
    results.append({
        'variant': entry.name,
        'steps': env.steps,
        'simulated_seconds': last.get('elapsed_seconds', 0.0),
        'reward': reward_total,
        'cumulative_component_total': component_total,
        'reward_contract': REWARD_CONTRACT_ID,
        'kills': last.get('total_kills', 0),
        'eva_attempts': last.get('eva_attempts', 0),
        'valid_eva_casts': last.get('valid_eva_casts', 0),
        'invalid_eva_attempts': last.get('invalid_eva_attempts', 0),
        'observation_shape': list(observation.shape),
        'vision_radius_cells': env.vision_radius_cells,
        'eva_radius_cells': env.eva_radius_cells,
    })
    env.close()
print(json.dumps(results, indent=2))
"@
if ($LASTEXITCODE -ne 0) { throw "v1.7 reward-audited smoke test failed." }

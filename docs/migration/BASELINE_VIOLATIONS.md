# Phase-1 Baseline Violations

> Generated deterministically by `docs/migration/tools/migration_integrity.py snapshot`.
> Do not normalize this debt away by hand. Later phases may deliberately shrink it; growth fails.

Base: `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`. Phase: 1.

## 1. ACCEPTED PRE-EXISTING MIGRATION DEBT

### R6 (7)

| concept/symbol | exact file | evidence |
|---|---|---|
| `observation_schema_id` | `flyff_farming_recorder/recorder/session.py` | `duplicate_definition=OBSERVATION_SCHEMA_ID` |
| `observation_schema_id` | `flyff_farming_simulator/farming/observation.py` | `duplicate_definition=OBSERVATION_SCHEMA_ID` |
| `observation_schema_id` | `foreground_vision_bot/farming/observation.py` | `duplicate_definition=OBSERVATION_SCHEMA_ID` |
| `observation_size` | `flyff_farming_simulator/farming/observation.py` | `duplicate_definition=OBSERVATION_SIZE` |
| `observation_size` | `foreground_vision_bot/farming/observation.py` | `duplicate_definition=OBSERVATION_SIZE` |
| `policy_action_nvecs` | `flyff_farming_simulator/farming/actions.py` | `duplicate_definition=POLICY_ACTION_NVECS` |
| `policy_action_nvecs` | `foreground_vision_bot/farming/actions.py` | `duplicate_definition=POLICY_ACTION_NVECS` |

### R7a (35)

| concept/symbol | exact file | evidence |
|---|---|---|
| `farming_action_contract` | `flyff_farming_simulator/farming/actions.py` | `duplicate_definition=FarmingAction` |
| `farming_action_contract` | `flyff_farming_simulator/farming/actions.py` | `duplicate_definition=FarmingEvent` |
| `farming_action_contract` | `flyff_farming_simulator/farming/actions.py` | `duplicate_definition=POLICY_ACTION_NVECS` |
| `farming_action_contract` | `flyff_farming_simulator/farming/actions.py` | `duplicate_definition=SteeringAction` |
| `farming_action_contract` | `foreground_vision_bot/farming/actions.py` | `duplicate_definition=FarmingAction` |
| `farming_action_contract` | `foreground_vision_bot/farming/actions.py` | `duplicate_definition=FarmingEvent` |
| `farming_action_contract` | `foreground_vision_bot/farming/actions.py` | `duplicate_definition=POLICY_ACTION_NVECS` |
| `farming_action_contract` | `foreground_vision_bot/farming/actions.py` | `duplicate_definition=SteeringAction` |
| `farming_model_contract` | `flyff_farming_simulator/farming/model_contract.py` | `duplicate_definition=CURRENT_MODEL_CONTRACT` |
| `farming_model_contract` | `flyff_farming_simulator/farming/model_contract.py` | `duplicate_definition=ModelContractMetadata` |
| `farming_model_contract` | `flyff_farming_simulator/farming/model_contract.py` | `duplicate_definition=validate_model_contract` |
| `farming_model_contract` | `foreground_vision_bot/farming/model_contract.py` | `duplicate_definition=CURRENT_MODEL_CONTRACT` |
| `farming_model_contract` | `foreground_vision_bot/farming/model_contract.py` | `duplicate_definition=ModelContractMetadata` |
| `farming_model_contract` | `foreground_vision_bot/farming/model_contract.py` | `duplicate_definition=validate_model_contract` |
| `farming_observation_contract` | `flyff_farming_recorder/recorder/session.py` | `duplicate_definition=OBSERVATION_SCHEMA_ID` |
| `farming_observation_contract` | `flyff_farming_simulator/farming/observation.py` | `duplicate_definition=OBSERVATION_SCHEMA_ID` |
| `farming_observation_contract` | `flyff_farming_simulator/farming/observation.py` | `duplicate_definition=OBSERVATION_SIZE` |
| `farming_observation_contract` | `flyff_farming_simulator/farming/observation.py` | `duplicate_definition=ObservationBuilder` |
| `farming_observation_contract` | `foreground_vision_bot/farming/observation.py` | `duplicate_definition=OBSERVATION_SCHEMA_ID` |
| `farming_observation_contract` | `foreground_vision_bot/farming/observation.py` | `duplicate_definition=OBSERVATION_SIZE` |
| `farming_observation_contract` | `foreground_vision_bot/farming/observation.py` | `duplicate_definition=ObservationBuilder` |
| `farming_reward_contract` | `flyff_farming_simulator/farming/reward.py` | `duplicate_definition=RewardCalculator` |
| `farming_reward_contract` | `flyff_farming_simulator/farming/reward.py` | `duplicate_definition=RewardComponents` |
| `farming_reward_contract` | `foreground_vision_bot/farming/reward.py` | `duplicate_definition=RewardCalculator` |
| `farming_reward_contract` | `foreground_vision_bot/farming/reward.py` | `duplicate_definition=RewardComponents` |
| `farming_session_contract` | `flyff_farming_simulator/farming/session.py` | `duplicate_definition=SessionEndReason` |
| `farming_session_contract` | `flyff_farming_simulator/farming/session.py` | `duplicate_definition=SessionOutcome` |
| `farming_session_contract` | `foreground_vision_bot/farming/session.py` | `duplicate_definition=SessionEndReason` |
| `farming_session_contract` | `foreground_vision_bot/farming/session.py` | `duplicate_definition=SessionOutcome` |
| `native_position_stack` | `flyff_farming_recorder/position/NativeFlyffMonsterProvider.py` | `duplicate_definition=NativeFlyffMonsterProvider` |
| `native_position_stack` | `flyff_farming_recorder/position/NativeFlyffPositionProvider.py` | `duplicate_definition=NativeFlyffPositionProvider` |
| `native_position_stack` | `flyff_farming_recorder/position/native_process_service.py` | `duplicate_definition=NativeProcessService` |
| `native_position_stack` | `foreground_vision_bot/position/NativeFlyffMonsterProvider.py` | `duplicate_definition=NativeFlyffMonsterProvider` |
| `native_position_stack` | `foreground_vision_bot/position/NativeFlyffPositionProvider.py` | `duplicate_definition=NativeFlyffPositionProvider` |
| `native_position_stack` | `foreground_vision_bot/position/native_process_service.py` | `duplicate_definition=NativeProcessService` |

### R7b (0)

| concept/symbol | exact file | evidence |
|---|---|---|
| _none_ | _none_ | rule currently green |

### R7c (20)

| concept/symbol | exact file | evidence |
|---|---|---|
| `CURRENT_MODEL_CONTRACT` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.model_contract:CURRENT_MODEL_CONTRACT` |
| `FarmingAction` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.actions:FarmingAction` |
| `FarmingEvent` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.actions:FarmingEvent` |
| `ModelContractMetadata` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.model_contract:ModelContractMetadata` |
| `NativeFlyffMonsterProvider` | `flyff_farming_recorder/position/__init__.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/position/__init__.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffPositionProvider` | `flyff_farming_recorder/position/__init__.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/position/__init__.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeProcessService` | `flyff_farming_recorder/position/__init__.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/__init__.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `OBSERVATION_SCHEMA_ID` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.observation:OBSERVATION_SCHEMA_ID` |
| `OBSERVATION_SIZE` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.observation:OBSERVATION_SIZE` |
| `ObservationBuilder` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.observation:ObservationBuilder` |
| `POLICY_ACTION_NVECS` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.actions:POLICY_ACTION_NVECS` |
| `RewardCalculator` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.reward:RewardCalculator` |
| `RewardComponents` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.reward:RewardComponents` |
| `SessionEndReason` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.session:SessionEndReason` |
| `SessionOutcome` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.session:SessionOutcome` |
| `SteeringAction` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.actions:SteeringAction` |
| `validate_model_contract` | `foreground_vision_bot/farming/__init__.py` | `reexport_from=.model_contract:validate_model_contract` |

## 2. RULES ALREADY GREEN

- R9: **GREEN** (0 violations).
- R10 policy-class modules: **GREEN** across 313 inventory rows.
- R10 full serialized references: **GREEN** across 317 reference rows.

## 3. DIAGNOSTIC-ONLY FINDINGS

- D1 exact SHA-256 pairs (tracked files >=200 bytes): **119**.
- D1 AST-normalized Python pairs >=95% similar: **31**.
- Exact deterministic evidence: `docs/migration/DUPLICATE_CONTENT_REPORT.tsv`.
- D1 never gates and did not trigger source deletion or merging.

## 4. UNRESOLVED/REQUIRES LATER PHASE

- R6/R7a farming ownership debt resolves during Phase 4 canonical farming.
- Native position ownership debt resolves during Phase 5 after its real-client gate.
- R7b remains active with current-layout semantics and tightens as legacy/archive boundaries appear.
- R7c baseline re-exports must shrink or become explicit registered shims; none was installed in Phase 1.

## 5. NEW BLOCKING VIOLATIONS

None at snapshot time.

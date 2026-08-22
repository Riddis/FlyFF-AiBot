# Phase-1 Baseline Violations

> Generated deterministically by `docs/migration/tools/migration_integrity.py snapshot`.
> Do not normalize this debt away by hand. Later phases may deliberately shrink it; growth fails.

Base: `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`. Phase: 4 (from `CANONICAL_OWNERS.toml`).

## 1. ACCEPTED PRE-EXISTING MIGRATION DEBT

### R6 (0)

| concept/symbol | exact file | evidence |
|---|---|---|
| _none_ | _none_ | rule currently green |

### R7a (6)

| concept/symbol | exact file | evidence |
|---|---|---|
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

### R7c (180)

| concept/symbol | exact file | evidence |
|---|---|---|
| `CURRENT_MODEL_CONTRACT` | `foreground_vision_bot/tests/test_farming_model_contract.py` | `reexport_from=farming.model_contract:CURRENT_MODEL_CONTRACT` |
| `FarmingAction` | `flyff_farming_simulator/farming/observation.py` | `reexport_from=.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/cli.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/demonstrations.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/environment.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/fair_time_cli.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/recording_discovery.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/scripted_policies.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/stagnation_diagnostics.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/simulator/steering_oracle.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_deep_review.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_environment_planner_kernel_agreement.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_factorized_actions_v19.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_fair_time_v16.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_fair_time_v17.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_physics_version_tag_provenance_only.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_reward_audit_v17.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `flyff_farming_simulator/tests/test_simulator_core.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `foreground_vision_bot/farming/control.py` | `reexport_from=.actions:FarmingAction` |
| `FarmingAction` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.actions:FarmingAction` |
| `FarmingAction` | `foreground_vision_bot/farming/trainer.py` | `reexport_from=.actions:FarmingAction` |
| `FarmingAction` | `foreground_vision_bot/tests/test_farming_actions.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingAction` | `foreground_vision_bot/tests/test_farming_observation_contract.py` | `reexport_from=farming.actions:FarmingAction` |
| `FarmingEvent` | `flyff_farming_simulator/scratchpad_monster_approach_baseline_eval.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/basic_training.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/dagger_v193.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/demonstrations.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/environment.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/factorized_cli.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/factorized_training.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/factorized_v193_cli.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/factorized_v193_training.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/milestone_evaluator.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/navigation_dataset.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/navigation_history.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/recovery_controller.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/simulator/scripted_policies.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/tests/test_dagger_v193.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/tests/test_factorized_actions_v19.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/tests/test_navigation_history.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/tests/test_recovery_controller.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `flyff_farming_simulator/tests/test_temporal_sidecar_parity.py` | `reexport_from=farming.actions:FarmingEvent` |
| `FarmingEvent` | `foreground_vision_bot/farming/control.py` | `reexport_from=.actions:FarmingEvent` |
| `FarmingEvent` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.actions:FarmingEvent` |
| `FarmingEvent` | `foreground_vision_bot/tests/test_factorized_control_v19.py` | `reexport_from=farming.actions:FarmingEvent` |
| `ModelContractMetadata` | `flyff_farming_simulator/simulator/cli.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/simulator/factorized_cli.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/simulator/factorized_training.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/simulator/factorized_v193_cli.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/simulator/fair_time_cli.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/simulator/training.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/tests/test_dagger_v193.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/tests/test_factorized_actions_v19.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/tests/test_resume_ppo_chunk.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `flyff_farming_simulator/tests/test_split_branch_policy.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `foreground_vision_bot/farming/reporting.py` | `reexport_from=.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `foreground_vision_bot/farming/startup.py` | `reexport_from=.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `foreground_vision_bot/tests/test_farming_model_contract.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `ModelContractMetadata` | `foreground_vision_bot/tests/test_farming_startup_reporting.py` | `reexport_from=farming.model_contract:ModelContractMetadata` |
| `NativeFlyffMonsterProvider` | `flyff_farming_recorder/position/__init__.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `flyff_farming_recorder/position/attachment_factory.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `flyff_farming_recorder/position/monster_factory.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/Bot.py` | `reexport_from=position:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/position/__init__.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/position/attachment_factory.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/position/monster_factory.py` | `reexport_from=.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/tests/test_native_monster_provider.py` | `reexport_from=position.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/tests/test_native_process_service.py` | `reexport_from=position.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `foreground_vision_bot/tests/test_pointer_recovery_stabilization.py` | `reexport_from=position.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffMonsterProvider` | `refactor_logs/profiles/runtime_native_pointer_harness.py` | `reexport_from=foreground_vision_bot.position.NativeFlyffMonsterProvider:NativeFlyffMonsterProvider` |
| `NativeFlyffPositionProvider` | `flyff_farming_recorder/position/__init__.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `flyff_farming_recorder/position/attachment_factory.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `flyff_farming_recorder/position/factory.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/position/__init__.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/position/attachment_factory.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/position/factory.py` | `reexport_from=.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/tests/test_native_position_provider.py` | `reexport_from=position.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/tests/test_native_process_service.py` | `reexport_from=position.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/tests/test_pointer_recovery_stabilization.py` | `reexport_from=position.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `foreground_vision_bot/tools/probe_native_position.py` | `reexport_from=position:NativeFlyffPositionProvider` |
| `NativeFlyffPositionProvider` | `refactor_logs/profiles/runtime_native_pointer_harness.py` | `reexport_from=foreground_vision_bot.position.NativeFlyffPositionProvider:NativeFlyffPositionProvider` |
| `NativeProcessService` | `flyff_farming_recorder/position/NativeFlyffMonsterProvider.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `flyff_farming_recorder/position/NativeFlyffPositionProvider.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `flyff_farming_recorder/position/__init__.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `flyff_farming_recorder/position/attachment_factory.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `flyff_farming_recorder/position/factory.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `flyff_farming_recorder/position/monster_factory.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/Bot.py` | `reexport_from=position:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/NativeFlyffMonsterProvider.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/NativeFlyffPositionProvider.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/__init__.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/attachment_factory.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/factory.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/position/monster_factory.py` | `reexport_from=.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/tests/test_anchored_pointer_recovery.py` | `reexport_from=position.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/tests/test_native_process_service.py` | `reexport_from=position.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/tests/test_pointer_persistence_transaction.py` | `reexport_from=position.native_process_service:NativeProcessService` |
| `NativeProcessService` | `foreground_vision_bot/tools/run_observation_telemetry.py` | `reexport_from=position.native_process_service:NativeProcessService` |
| `OBSERVATION_SCHEMA_ID` | `flyff_farming_simulator/farming/model_contract.py` | `reexport_from=.observation:OBSERVATION_SCHEMA_ID` |
| `OBSERVATION_SCHEMA_ID` | `flyff_farming_simulator/simulator/cli.py` | `reexport_from=farming.observation:OBSERVATION_SCHEMA_ID` |
| `OBSERVATION_SCHEMA_ID` | `flyff_farming_simulator/simulator/demonstrations.py` | `reexport_from=farming.observation:OBSERVATION_SCHEMA_ID` |
| `OBSERVATION_SCHEMA_ID` | `foreground_vision_bot/tests/test_farming_observation_contract.py` | `reexport_from=farming.observation:OBSERVATION_SCHEMA_ID` |
| `OBSERVATION_SIZE` | `flyff_farming_simulator/farming/model_contract.py` | `reexport_from=.observation:OBSERVATION_SIZE` |
| `OBSERVATION_SIZE` | `foreground_vision_bot/farming/sb3_adapter.py` | `reexport_from=.observation:OBSERVATION_SIZE` |
| `OBSERVATION_SIZE` | `foreground_vision_bot/tests/test_farming_observation_contract.py` | `reexport_from=farming.observation:OBSERVATION_SIZE` |
| `ObservationBuilder` | `flyff_farming_simulator/simulator/demonstrations.py` | `reexport_from=farming.observation:ObservationBuilder` |
| `ObservationBuilder` | `flyff_farming_simulator/simulator/environment.py` | `reexport_from=farming.observation:ObservationBuilder` |
| `ObservationBuilder` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.observation:ObservationBuilder` |
| `ObservationBuilder` | `foreground_vision_bot/tests/test_farming_observation_contract.py` | `reexport_from=farming.observation:ObservationBuilder` |
| `POLICY_ACTION_NVECS` | `flyff_farming_simulator/farming/model_contract.py` | `reexport_from=.actions:POLICY_ACTION_NVECS` |
| `POLICY_ACTION_NVECS` | `flyff_farming_simulator/simulator/demonstrations.py` | `reexport_from=farming.actions:POLICY_ACTION_NVECS` |
| `POLICY_ACTION_NVECS` | `flyff_farming_simulator/simulator/environment.py` | `reexport_from=farming.actions:POLICY_ACTION_NVECS` |
| `POLICY_ACTION_NVECS` | `flyff_farming_simulator/simulator/factorized_cli.py` | `reexport_from=farming.actions:POLICY_ACTION_NVECS` |
| `POLICY_ACTION_NVECS` | `flyff_farming_simulator/simulator/factorized_v193_cli.py` | `reexport_from=farming.actions:POLICY_ACTION_NVECS` |
| `POLICY_ACTION_NVECS` | `foreground_vision_bot/farming/sb3_adapter.py` | `reexport_from=.actions:POLICY_ACTION_NVECS` |
| `RewardCalculator` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.reward:RewardCalculator` |
| `RewardCalculator` | `foreground_vision_bot/tests/test_farming_reward_and_session.py` | `reexport_from=farming.reward:RewardCalculator` |
| `SessionEndReason` | `flyff_farming_simulator/farming/reward.py` | `reexport_from=.session:SessionEndReason` |
| `SessionEndReason` | `flyff_farming_simulator/simulator/reward_model.py` | `reexport_from=farming.session:SessionEndReason` |
| `SessionEndReason` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.session:SessionEndReason` |
| `SessionEndReason` | `foreground_vision_bot/tests/test_farming_reward_and_session.py` | `reexport_from=farming.session:SessionEndReason` |
| `SessionOutcome` | `flyff_farming_simulator/farming/reward.py` | `reexport_from=.session:SessionOutcome` |
| `SessionOutcome` | `flyff_farming_simulator/simulator/environment.py` | `reexport_from=farming.session:SessionOutcome` |
| `SessionOutcome` | `flyff_farming_simulator/simulator/reward_model.py` | `reexport_from=farming.session:SessionOutcome` |
| `SessionOutcome` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.session:SessionOutcome` |
| `SessionOutcome` | `foreground_vision_bot/tests/test_farming_reward_and_session.py` | `reexport_from=farming.session:SessionOutcome` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/simulator/factorized_v193_cli.py` | `reexport_from=.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/tests/test_dagger_v193.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/tests/test_milestone_evaluator.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/tests/test_milestone_evaluator_recovery.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/tests/test_resume_ppo_chunk.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/tests/test_split_branch_policy.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringEventPolicy` | `flyff_farming_simulator/tests/test_steering_expansion_transplant.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringEventPolicy` |
| `SplitSteeringNavigationPolicy` | `flyff_farming_simulator/scratchpad_ppo_pure_navigation.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy` |
| `SplitSteeringNavigationPolicy` | `flyff_farming_simulator/scratchpad_ppo_pure_navigation_v2.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy` |
| `SplitSteeringNavigationPolicy` | `flyff_farming_simulator/tests/test_fine_tune_steering_branch.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy` |
| `SplitSteeringNavigationPolicy` | `flyff_farming_simulator/tests/test_steering_expansion_transplant.py` | `reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy` |
| `SteeringAction` | `flyff_farming_simulator/scratchpad_coarse_route_proof_of_mechanism_v2.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/scratchpad_coarse_route_rollout_verification.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/scratchpad_debug_waypoint_no_effect.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/scratchpad_diagnose_v3_terminal_gate_onsets.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/scratchpad_ppo_pure_navigation_v2.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/basic_training.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/demonstrations.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/environment.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/factorized_cli.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/factorized_training.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/factorized_v193_cli.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/factorized_v193_training.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/recovery_controller.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/scripted_policies.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/stagnation_diagnostics.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/simulator/steering_oracle.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/tests/test_factorized_actions_v19.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/tests/test_navigation_history.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/tests/test_recovery_controller.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/tests/test_steering_oracle_escape_robust.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/tests/test_steering_oracle_v3_terminal_gate.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `flyff_farming_simulator/tests/test_temporal_sidecar_parity.py` | `reexport_from=farming.actions:SteeringAction` |
| `SteeringAction` | `foreground_vision_bot/farming/environment.py` | `reexport_from=.actions:SteeringAction` |
| `SteeringAction` | `foreground_vision_bot/tests/test_factorized_control_v19.py` | `reexport_from=farming.actions:SteeringAction` |
| `TerminalPrefixRolloutBuffer` | `foreground_vision_bot/farming/trainer.py` | `reexport_from=.sb3_training:TerminalPrefixRolloutBuffer` |
| `TerminalPrefixRolloutBuffer` | `foreground_vision_bot/tests/test_farming_sb3_training.py` | `reexport_from=farming.sb3_training:TerminalPrefixRolloutBuffer` |
| `TrainingBoundary` | `foreground_vision_bot/tests/test_farming_training_session.py` | `reexport_from=farming.sb3_training:TrainingBoundary` |
| `TrainingBoundaryKind` | `foreground_vision_bot/farming/trainer.py` | `reexport_from=.sb3_training:TrainingBoundaryKind` |
| `TrainingBoundaryKind` | `foreground_vision_bot/tests/test_farming_sb3_training.py` | `reexport_from=farming.sb3_training:TrainingBoundaryKind` |
| `TrainingBoundaryKind` | `foreground_vision_bot/tests/test_farming_training_session.py` | `reexport_from=farming.sb3_training:TrainingBoundaryKind` |
| `select_persistent_waypoint` | `flyff_farming_simulator/scratchpad_audit_selector_fallback.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/scratchpad_diagnose_ltr_rtl_contrastive.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/scratchpad_diagnose_two_wall_rtl_67_regression.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/scratchpad_general_router_episode.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/scratchpad_monster_approach_baseline_eval.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/scratchpad_promotion_equivalence_check.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/simulator/router_waypoint_env.py` | `reexport_from=.kinodynamic_route_planner:select_persistent_waypoint` |
| `select_persistent_waypoint` | `flyff_farming_simulator/tests/test_kinodynamic_route_planner.py` | `reexport_from=simulator.kinodynamic_route_planner:select_persistent_waypoint` |
| `validate_model_contract` | `flyff_farming_simulator/simulator/factorized_training.py` | `reexport_from=farming.model_contract:validate_model_contract` |
| `validate_model_contract` | `flyff_farming_simulator/simulator/training.py` | `reexport_from=farming.model_contract:validate_model_contract` |
| `validate_model_contract` | `flyff_farming_simulator/tests/test_factorized_actions_v19.py` | `reexport_from=farming.model_contract:validate_model_contract` |
| `validate_model_contract` | `foreground_vision_bot/farming/startup.py` | `reexport_from=.model_contract:validate_model_contract` |
| `validate_model_contract` | `foreground_vision_bot/tests/test_farming_model_contract.py` | `reexport_from=farming.model_contract:validate_model_contract` |

## 2. RULES ALREADY GREEN

- R9: **GREEN** (0 violations).
- R10 policy-class modules: **GREEN** across 313 inventory rows.
- R10 full serialized references: **GREEN** across 317 reference rows.
- R10 module classification: `{"farming.sb3_training": "repository-local", "simulator.split_branch_policy": "repository-local", "stable_baselines3.common.policies": "external"}`.
- Phase-1 R10 protects the frozen Phase-0 checkpoint corpus described by `CHECKPOINT_INVENTORY.tsv` and `CHECKPOINT_MODULE_REFERENCES.tsv`.
- Phase-2 G10a independently regenerates that corpus inventory against preserved artifacts.

## 3. DIAGNOSTIC-ONLY FINDINGS

- D1 exact SHA-256 pairs (tracked files >=200 bytes): **114**.
- D1 AST-normalized Python pairs >=95% similar: **25**.
- Exact deterministic evidence: `docs/migration/DUPLICATE_CONTENT_REPORT.tsv`.
- D1 never gates and did not trigger source deletion or merging.

## 4. UNRESOLVED/REQUIRES LATER PHASE

- Shared farming ownership is canonical under `flyff_farming_simulator/farming`; R6 and farming R7a are green.
- Native position ownership debt resolves during Phase 5 after its real-client gate.
- R7b remains active with current-layout semantics and tightens as legacy/archive boundaries appear.
- R7c contains current repository re-exports; every Phase-4 B1 re-export is explicitly registered through Phase 6.

## 5. NEW BLOCKING VIOLATIONS

None at snapshot time.

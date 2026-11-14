# Test Report

Generated: 2026-11-14 23:13 by `python scripts/run_tests.py`

## Summary

| Total | Passed | Failed | Skipped | Duration (s) |
|---|---|---|---|---|
| 74 | 74 | 0 | 0 | 13.2 |

## Results by module

| Module | Tests | Passed | Failed |
|---|---|---|---|
| test_api | 26 | 26 | 0 |
| test_core | 19 | 19 | 0 |
| test_data_pipeline | 13 | 13 | 0 |
| test_modeling | 16 | 16 | 0 |

## Acceptance criteria

| Criterion | Required | Actual | Result |
|---|---|---|---|
| All automated tests pass | 100 % | 100 % | PASS |
| Model roc_auc on test set | >= 0.78 | 0.8341 | PASS |
| Model recall on test set | >= 0.7 | 0.8176 | PASS |
| Model precision on test set | >= 0.25 | 0.3011 | PASS |

## All test cases

| Module | Test | Result | Time (s) |
|---|---|---|---|
| test_api | `test_health` | passed | 0.41 |
| test_api | `test_ready` | passed | 0.02 |
| test_api | `test_predict_valid_request` | passed | 0.03 |
| test_api | `test_healthy_person_scores_lower_than_high_risk_person` | passed | 0.04 |
| test_api | `test_invalid_values_rejected[BMI-5]` | passed | 0.02 |
| test_api | `test_invalid_values_rejected[Age-14]` | passed | 0.02 |
| test_api | `test_invalid_values_rejected[HighBP-2]` | passed | 0.01 |
| test_api | `test_invalid_values_rejected[GenHlth-0]` | passed | 0.02 |
| test_api | `test_invalid_values_rejected[MentHlth-31]` | passed | 0.01 |
| test_api | `test_invalid_values_rejected[Sex-male]` | passed | 0.02 |
| test_api | `test_invalid_values_rejected[Income-2.5]` | passed | 0.02 |
| test_api | `test_missing_field_rejected` | passed | 0.02 |
| test_api | `test_unknown_field_rejected` | passed | 0.01 |
| test_api | `test_malformed_json_rejected` | passed | 0.02 |
| test_api | `test_request_id_echoed_and_generated` | passed | 0.04 |
| test_api | `test_model_unavailable_returns_503` | passed | 0.02 |
| test_api | `test_unexpected_error_returns_generic_500` | passed | 0.04 |
| test_api | `test_metrics_endpoint_exposes_monitoring` | passed | 0.04 |
| test_api | `test_predictions_are_logged_encrypted` | passed | 0.04 |
| test_api | `test_drift_endpoint` | passed | 0.38 |
| test_api | `test_pipeline_endpoint_structure` | passed | 0.03 |
| test_api | `test_model_info` | passed | 0.01 |
| test_api | `test_reload_requires_admin_token` | passed | 0.02 |
| test_api | `test_mlflow_predictor_parses_response` | passed | 0.01 |
| test_api | `test_mlflow_predictor_connection_error` | passed | 0.01 |
| test_api | `test_invalid_inference_mode_rejected` | passed | 0.01 |
| test_core | `test_schema_has_21_unique_features` | passed | 0.00 |
| test_core | `test_healthy_reference_values_are_within_range` | passed | 0.00 |
| test_core | `test_risk_level_bands_follow_threshold[0.0-low]` | passed | 0.00 |
| test_core | `test_risk_level_bands_follow_threshold[0.29-low]` | passed | 0.00 |
| test_core | `test_risk_level_bands_follow_threshold[0.3-moderate]` | passed | 0.00 |
| test_core | `test_risk_level_bands_follow_threshold[0.49-moderate]` | passed | 0.00 |
| test_core | `test_risk_level_bands_follow_threshold[0.5-high]` | passed | 0.01 |
| test_core | `test_risk_level_bands_follow_threshold[0.99-high]` | passed | 0.01 |
| test_core | `test_add_features_appends_engineered_columns` | passed | 0.02 |
| test_core | `test_feature_values_are_correct` | passed | 0.01 |
| test_core | `test_add_features_does_not_modify_input` | passed | 0.02 |
| test_core | `test_encrypted_log_roundtrip` | passed | 0.01 |
| test_core | `test_wrong_key_cannot_read_log` | passed | 0.00 |
| test_core | `test_no_key_means_no_cipher` | passed | 0.00 |
| test_core | `test_track_stage_records_success` | passed | 0.01 |
| test_core | `test_track_stage_records_failure_and_reraises` | passed | 0.03 |
| test_core | `test_read_runs_newest_first_and_skips_corrupt_lines` | passed | 0.00 |
| test_core | `test_dvc_dag_edges_match_pipeline` | passed | 0.01 |
| test_core | `test_dag_falls_back_when_dvc_file_missing` | passed | 0.00 |
| test_data_pipeline | `test_invalid_mask_flags_each_kind_of_problem` | passed | 0.04 |
| test_data_pipeline | `test_invalid_mask_accepts_arrow_dtypes` | passed | 0.03 |
| test_data_pipeline | `test_non_numeric_values_are_invalid` | passed | 0.03 |
| test_data_pipeline | `test_batch_quality_counts` | passed | 0.05 |
| test_data_pipeline | `test_clean_batch_drops_invalid_rows_and_casts` | passed | 0.04 |
| test_data_pipeline | `test_quality_gate[q0-True]` | passed | 0.00 |
| test_data_pipeline | `test_quality_gate[q1-False]` | passed | 0.00 |
| test_data_pipeline | `test_quality_gate[q2-False]` | passed | 0.00 |
| test_data_pipeline | `test_quality_gate[q3-False]` | passed | 0.00 |
| test_data_pipeline | `test_split_is_reproducible_regardless_of_row_order` | passed | 0.09 |
| test_data_pipeline | `test_ingest_synthetic_source_has_schema` | passed | 0.01 |
| test_data_pipeline | `test_ingest_rejects_unknown_source` | passed | 0.00 |
| test_data_pipeline | `test_ray_pipeline_end_to_end` | passed | 10.37 |
| test_modeling | `test_each_model_type_trains_and_predicts[logistic_regression-hp0]` | passed | 0.03 |
| test_modeling | `test_each_model_type_trains_and_predicts[random_forest-hp1]` | passed | 0.23 |
| test_modeling | `test_each_model_type_trains_and_predicts[xgboost-hp2]` | passed | 0.15 |
| test_modeling | `test_unknown_model_type_raises` | passed | 0.00 |
| test_modeling | `test_higher_beta_gives_higher_recall` | passed | 0.44 |
| test_modeling | `test_metrics_contain_required_keys` | passed | 0.03 |
| test_modeling | `test_feature_importance_is_normalised` | passed | 0.17 |
| test_modeling | `test_acceptance_passes_and_fails` | passed | 0.00 |
| test_modeling | `test_registration_retries_after_connection_error` | passed | 0.01 |
| test_modeling | `test_psi_is_zero_for_identical_distributions` | passed | 0.01 |
| test_modeling | `test_no_drift_on_same_distribution` | passed | 0.01 |
| test_modeling | `test_drift_detected_when_bmi_shifts` | passed | 0.01 |
| test_modeling | `test_drift_needs_minimum_samples` | passed | 0.01 |
| test_modeling | `test_what_if_only_changes_modifiable_unhealthy_factors` | passed | 0.00 |
| test_modeling | `test_low_bmi_is_never_pushed_up` | passed | 0.00 |
| test_modeling | `test_top_factors_sorted_and_filtered` | passed | 0.00 |

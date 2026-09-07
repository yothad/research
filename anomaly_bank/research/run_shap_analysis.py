"""SHAP feature-importance sanity check for both fraudTest.csv tracks, on the
current (pre-Phase-2) feature set. Quick diagnostic, not a full tuning pass —
see the session discussion: full optimization belongs after Phase 2 adds the
network-velocity features.

    python research/run_shap_analysis.py
"""
import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess
import steps.fraud_test_isolation_forest as fraud_test_isolation_forest
import steps.xgboost as xgboost_step
import utils.models as models

SAMPLE_SIZE = 2000

if __name__ == '__main__':
    pre = fraud_test_preprocess.FraudTestPreprocess(config_path=bootstrap.CONFIG_PATH)
    raw_df, enriched_df, feat_columns = pre._execute()
    target_col = pre.cfg['target_col']

    # --- XGBoost: the main track ---
    xgb_obj = xgboost_step.FraudTestXGBoostModel(config_path=bootstrap.CONFIG_PATH, enriched_df=enriched_df, feat_columns=feat_columns)
    X_train, X_test, y_train, y_test = xgb_obj.split(target_col)
    xgb_obj.fit(X_train, y_train)

    xgb_sample = X_test.sample(min(SAMPLE_SIZE, len(X_test)), random_state=42)
    xgb_importance = models.compute_shap_values(xgb_obj.model, xgb_sample, feat_columns)
    print("=== XGBoost SHAP importance (mean |shap|) ===")
    print(xgb_importance.to_string(index=False))

    # --- Isolation Forest: comparison / cheap second opinion ---
    if_obj = fraud_test_isolation_forest.FraudTestIsolationForestModel(config_path=bootstrap.CONFIG_PATH, enriched_df=enriched_df, feat_columns=feat_columns)
    if_obj.run_model()

    if_sample = enriched_df[feat_columns].fillna(0).sample(min(SAMPLE_SIZE, len(enriched_df)), random_state=42)
    if_importance = models.compute_shap_values(if_obj.model, if_sample, feat_columns)
    print("\n=== Isolation Forest SHAP importance (mean |shap|) ===")
    print(if_importance.to_string(index=False))

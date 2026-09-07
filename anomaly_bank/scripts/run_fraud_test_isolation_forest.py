"""Run the lean Isolation Forest track on fraudTest.csv and evaluate against the real label.

    python scripts/run_isolation_forest.py
"""
import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess
import steps.fraud_test_isolation_forest as fraud_test_isolation_forest

if __name__ == '__main__':
    pre = fraud_test_preprocess.FraudTestPreprocess(config_path=bootstrap.CONFIG_PATH)
    raw_df, enriched_df, feat_columns = pre._execute()
    target_col = pre.cfg['target_col']

    model_obj = fraud_test_isolation_forest.FraudTestIsolationForestModel(
        config_path=bootstrap.CONFIG_PATH, enriched_df=enriched_df, feat_columns=feat_columns
    )
    isolation_df, metrics = model_obj._execute(target_col=target_col)

    print(f"\nfeat_columns ({len(feat_columns)}): {feat_columns}")
    print("\nconfusion matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nprecision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"pr_auc:    {metrics['pr_auc']:.4f}")
    print(f"n_flagged: {metrics['n_flagged']}  (n_actual_fraud: {metrics['n_actual_fraud']})")

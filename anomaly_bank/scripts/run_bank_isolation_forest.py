"""Run the full Isolation Forest pipeline on bank_transactions_data.csv — this
dataset has no real fraud label, so this is the SHAP-ranked feature sweep +
per-row explainability (TopReasons) + tree-vote confidence version. Recreates
what the old run.ipynb notebook did, as a script.

    python scripts/run_bank_isolation_forest.py
"""
import _bootstrap as bootstrap
import steps.bank_isolation_forest as bank_isolation_forest

if __name__ == '__main__':
    pre = bank_isolation_forest.BankIsolationForestPreprocess(config_path=bootstrap.CONFIG_PATH)
    raw_data_df, enriched_df, feat_columns = pre._execute()

    model_obj = bank_isolation_forest.BankIsolationForestModel(
        config_path=bootstrap.CONFIG_PATH, raw_df=raw_data_df, enriched_df=enriched_df, feat_columns=feat_columns
    )
    anomaly_aggs_df, isolation_df, labeled_raw_df, shap_explanation = model_obj._execute()

    print(f"\nraw_data_df shape: {raw_data_df.shape}")
    print(f"best features ({len(model_obj.best_features)}): {model_obj.best_features}")

    print("\nIsolationForest_Anomaly value counts:")
    print(labeled_raw_df['IsolationForest_Anomaly'].value_counts())

    anomalies = labeled_raw_df[labeled_raw_df['IsolationForest_Anomaly'] == 1][
        ['TransactionID', 'IsolationForest_AnomalyScore', 'IsolationForest_Confidence',
         'IsolationForest_ConfidenceWithinAnomalies', 'IsolationForest_TopReasons']
    ].sort_values('IsolationForest_ConfidenceWithinAnomalies', ascending=False)
    print(f"\nFlagged anomalies ({len(anomalies)}):")
    print(anomalies.to_string(index=False))

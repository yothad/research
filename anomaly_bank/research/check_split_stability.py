"""Check how stable the XGBoost metrics are across different train/test splits.
Fraud is only 0.386% of the data, so a single 80/20 split's test set holds only
a few hundred fraud examples — precision/recall/PR-AUC could shift meaningfully
by chance alone. This runs stratified K-fold and reports mean +/- std.

    python scripts/check_split_stability.py
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess
import steps.xgboost as xgboost_step

N_FOLDS = 5

if __name__ == '__main__':
    pre = fraud_test_preprocess.FraudTestPreprocess(config_path=bootstrap.CONFIG_PATH)
    raw_df, enriched_df, feat_columns = pre._execute()
    target_col = pre.cfg['target_col']

    X = enriched_df[feat_columns]
    y = enriched_df[target_col]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    precisions, recalls, pr_aucs = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_obj = xgboost_step.FraudTestXGBoostModel(config_path=bootstrap.CONFIG_PATH, enriched_df=enriched_df, feat_columns=feat_columns)
        model_obj.fit(X_train, y_train)
        metrics = model_obj.evaluate(X_test, y_test)

        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        pr_aucs.append(metrics['pr_auc'])
        print(f"fold {fold}: precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  pr_auc={metrics['pr_auc']:.4f}  n_fraud_test={int(y_test.sum())}")

    print(f"\nprecision: {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
    print(f"recall:    {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
    print(f"pr_auc:    {np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}")

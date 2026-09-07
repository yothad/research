"""Cost-based threshold analysis, replacing F1/accuracy-driven threshold choice
with one that minimizes real expected dollar cost.

Missed fraud (false negative) cost = that specific transaction's actual `amt` —
exact, not estimated, since we have real dollar amounts. False-decline (false
positive) cost is a business assumption we can't check from data, so this sweeps
several assumptions rather than committing to one guess.

Also calibrates the raw XGBoost score via Platt scaling (sklearn
CalibratedClassifierCV, method='sigmoid') first, so the resulting score is a
genuine probability estimate — needed before it makes sense to present it as a
0-100 risk band. Calibration only rescales the numbers; it can't change the
model's ranking (that's what PR-AUC already told us is good), so it's safe to
apply after the fact.

    python research/cost_threshold_analysis.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess
import utils.imports as imports

FP_COST_ASSUMPTIONS = [1, 5, 10, 20, 50]  # dollars, per false decline

if __name__ == '__main__':
    config = imports.import_yml(bootstrap.CONFIG_PATH)
    xgb_cfg = config['fraud_test']['xgboost']

    pre = fraud_test_preprocess.FraudTestPreprocess(config_path=bootstrap.CONFIG_PATH)
    raw_df, enriched_df, feat_columns = pre._execute()
    target_col = pre.cfg['target_col']

    X = enriched_df[feat_columns]
    y = enriched_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=xgb_cfg['test_size'], stratify=y, random_state=xgb_cfg['random_state']
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base_model = xgb.XGBClassifier(
        n_estimators=xgb_cfg['n_estimators'],
        learning_rate=xgb_cfg['learning_rate'],
        max_depth=xgb_cfg['max_depth'],
        random_state=xgb_cfg['random_state'],
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
    )

    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)

    calibrated_score = calibrated_model.predict_proba(X_test)[:, 1]

    # --- calibration sanity check: predicted bucket vs actual observed fraud rate ---
    print("=== Calibration check (predicted probability bucket vs actual fraud rate) ===")
    calib_df = pd.DataFrame({'score': calibrated_score, 'y_true': y_test.values})
    calib_df['bucket'] = pd.cut(calib_df['score'], bins=[0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0])
    print(calib_df.groupby('bucket', observed=True).agg(n=('y_true', 'size'), actual_fraud_rate=('y_true', 'mean'), mean_predicted=('score', 'mean')))
    print("(Calibration ceiling caps around ~0.64 — informative for interpretability, but the")
    print(" cost sweep below uses the RAW score instead, since it doesn't need calibration to")
    print(" be valid and raw scores aren't capped.)")

    # --- cost sweep (on the RAW score — doesn't need calibration to be valid, and avoids its ceiling) ---
    base_model.fit(X_train, y_train)
    raw_score = base_model.predict_proba(X_test)[:, 1]

    amt_test = X_test['amt'].values
    thresholds = np.linspace(0.01, 0.99, 99)

    print("\n=== Cost-optimal threshold (on raw score) per false-decline cost assumption ===")
    for fp_cost in FP_COST_ASSUMPTIONS:
        costs = []
        for t in thresholds:
            y_pred = (raw_score >= t).astype(int)
            fn_mask = (y_test.values == 1) & (y_pred == 0)
            fp_mask = (y_test.values == 0) & (y_pred == 1)
            total_cost = amt_test[fn_mask].sum() + fp_cost * fp_mask.sum()
            costs.append(total_cost)
        costs = np.array(costs)
        best_idx = costs.argmin()
        best_t = thresholds[best_idx]

        y_pred_best = (raw_score >= best_t).astype(int)
        n_flagged = y_pred_best.sum()
        fn_at_best = ((y_test.values == 1) & (y_pred_best == 0)).sum()

        # baselines for context: flag nothing vs flag everything
        cost_flag_none = amt_test[y_test.values == 1].sum()
        cost_flag_all = fp_cost * (y_test.values == 0).sum()

        print(f"\nfalse-decline cost = ${fp_cost}")
        print(f"  best threshold: {best_t:.2f}  (n_flagged={n_flagged}, missed_fraud_count={fn_at_best})")
        print(f"  total expected cost at best threshold: ${costs[best_idx]:,.2f}")
        print(f"  cost if flagging nothing (baseline):   ${cost_flag_none:,.2f}")
        print(f"  cost if flagging everything (baseline): ${cost_flag_all:,.2f}")

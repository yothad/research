"""Run the full pipeline (preprocess -> XGBoost -> 3-tier routing) and report
the resulting decision distribution and dollar breakdown on the test set.
Policy reasoning: research/cost_model_findings.md.

    python scripts/run_risk_routing.py
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split

import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess
import steps.risk_routing as risk_routing
import utils.imports as imports

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
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg['n_estimators'], learning_rate=xgb_cfg['learning_rate'],
        max_depth=xgb_cfg['max_depth'], random_state=xgb_cfg['random_state'],
        scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]

    router = risk_routing.RiskRouter(config_path=bootstrap.CONFIG_PATH)
    decisions = router.route(score)

    amt = X_test['amt'].values
    y_true = y_test.values

    print("Decision distribution:")
    print(decisions.value_counts())

    print("\nBy decision x actual label (count / dollar value):")
    for decision in [risk_routing.AUTO_PASS, risk_routing.HUMAN_AUTH, risk_routing.AUTO_DECLINE]:
        mask = (decisions == decision).values
        for label, name in [(1, 'fraud'), (0, 'legit')]:
            sub_mask = mask & (y_true == label)
            print(f"  {decision:<14} x {name:<5}: n={sub_mask.sum():5d}  ${amt[sub_mask].sum():>12,.2f}")

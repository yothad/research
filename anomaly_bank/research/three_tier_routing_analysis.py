"""Cost-optimal thresholds for a 3-tier routing policy: Auto Pass / Human
Authentication / Auto Decline, instead of a single binary threshold.

Compares three ways of modeling the false-decline cost (base_friction, amt):
  flat:    base                          — ignores transaction size entirely
  capped:  base + AMT_FRACTION*min(amt, CAP)  — scales with size, but one huge
           outlier transaction (this dataset goes up to $22,768) can't dominate
  log:     base + LOG_K*log(1+amt)       — scales with size, heavily compressed
           for large values

A pure flat cost was too indifferent to size (26% of legit transactions here are
worth under $10, so a flat $10 exceeded the transaction's own value). A pure
linear fraction*amt swung too far the other way — 20% of a $22,768 outlier is
$4,553, making the optimizer extremely reluctant to decline anything, tripling
missed fraud. capped and log are two different ways of getting size-awareness
without that outlier sensitivity.

catch_rate is the share of real fraud stopped by a human-auth challenge itself
(assumption). auth_friction_cost is a fraction of the same per-row decline cost.

CONCLUSION (full reasoning trail in research/cost_model_findings.md):
`log` model won the flat/capped/log comparison. `base` was picked via an elbow
search (max curvature point on the missed-fraud-$ vs. legit-friction-$ tradeoff
curve) rather than eyeballed — landed on base~=$21.54, rounded to $20. Final
policy, formalized in steps/risk_routing.py: pass<0.69, human_auth<0.95, decline
>=0.95 on the raw XGBoost score.

    python research/three_tier_routing_analysis.py
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess
import utils.imports as imports

BASE_FRICTION_ASSUMPTIONS = [5, 10, 20]
AMT_FRACTION = 0.2      # capped model: + this fraction of min(amt, CAP)
CAP = 200                # capped model: amt contribution stops growing past this
LOG_K = 3                 # log model: + this * log(1+amt)
AUTH_FRICTION_FRACTION = 0.2
CATCH_RATE = 0.85
N_THRESHOLDS = 50


def fp_decline_cost_fn(model_name, base, amt):
    if model_name == 'flat':
        return np.full_like(amt, base, dtype=float)
    if model_name == 'capped':
        return base + AMT_FRACTION * np.minimum(amt, CAP)
    if model_name == 'log':
        return base + LOG_K * np.log1p(amt)
    raise ValueError(model_name)


def search_best_thresholds(score, y_true, amt, fp_decline_cost, thresholds):
    auth_friction_cost = AUTH_FRICTION_FRACTION * fp_decline_cost
    best_cost, best_pair = None, None
    for low_t in thresholds:
        for high_t in thresholds:
            if high_t <= low_t:
                continue
            pass_mask = score < low_t
            decline_mask = score >= high_t
            auth_mask = ~pass_mask & ~decline_mask

            cost = (
                amt[pass_mask & (y_true == 1)].sum()
                + fp_decline_cost[decline_mask & (y_true == 0)].sum()
                + (1 - CATCH_RATE) * amt[auth_mask & (y_true == 1)].sum()
                + auth_friction_cost[auth_mask & (y_true == 0)].sum()
            )
            if best_cost is None or cost < best_cost:
                best_cost, best_pair = cost, (low_t, high_t)
    return best_pair, best_cost


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

    amt = X_test['amt'].values
    y_true = y_test.values
    thresholds = np.linspace(0.01, 0.99, N_THRESHOLDS)

    # --- elbow search: sweep `base` finely for the log model, trace the
    # (missed_fraud_$, legit_friction_$) tradeoff curve, find where it bends ---
    print("########## log: elbow search across a fine base sweep ##########")
    fine_bases = np.geomspace(1, 100, 25)
    curve = []
    for base in fine_bases:
        fp_decline_cost = fp_decline_cost_fn('log', base, amt)
        (low_t, high_t), best_cost = search_best_thresholds(score, y_true, amt, fp_decline_cost, thresholds)
        pass_mask = score < low_t
        decline_mask = score >= high_t
        auth_mask = ~pass_mask & ~decline_mask
        missed_fraud_dollars = amt[pass_mask & (y_true == 1)].sum()
        legit_friction_dollars = amt[(decline_mask | auth_mask) & (y_true == 0)].sum()
        curve.append((base, missed_fraud_dollars, legit_friction_dollars, low_t, high_t))

    curve_arr = np.array([(c[1], c[2]) for c in curve])  # (missed_fraud, legit_friction)
    # normalize both axes to 0-1 so neither dollar scale dominates the elbow geometry
    x = (curve_arr[:, 0] - curve_arr[:, 0].min()) / (np.ptp(curve_arr[:, 0]) or 1)
    y_ = (curve_arr[:, 1] - curve_arr[:, 1].min()) / (np.ptp(curve_arr[:, 1]) or 1)
    # elbow = point with max perpendicular distance from the line connecting the two endpoints
    p1, p2 = np.array([x[0], y_[0]]), np.array([x[-1], y_[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y_[i]]) - p1
        proj = np.dot(p, line_vec_norm) * line_vec_norm
        distances.append(np.linalg.norm(p - proj))
    elbow_idx = int(np.argmax(distances))

    for i, (base, missed, friction, low_t, high_t) in enumerate(curve):
        marker = "  <-- elbow" if i == elbow_idx else ""
        print(f"base=${base:6.2f}  missed_fraud=${missed:9,.2f}  legit_friction=${friction:10,.2f}  "
              f"thresholds=[{low_t:.2f},{high_t:.2f}]{marker}")
    print()

    for model_name in ['flat', 'capped', 'log']:
        print(f"########## {model_name} ##########")
        for base in BASE_FRICTION_ASSUMPTIONS:
            fp_decline_cost = fp_decline_cost_fn(model_name, base, amt)
            (low_t, high_t), best_cost = search_best_thresholds(score, y_true, amt, fp_decline_cost, thresholds)

            pass_mask = score < low_t
            decline_mask = score >= high_t
            auth_mask = ~pass_mask & ~decline_mask

            missed_fraud_mask = pass_mask & (y_true == 1)
            fraud_to_auth_mask = auth_mask & (y_true == 1)
            fraud_declined_mask = decline_mask & (y_true == 1)
            legit_declined_mask = decline_mask & (y_true == 0)
            legit_to_auth_mask = auth_mask & (y_true == 0)

            print(f"base=${base}: thresholds pass<{low_t:.2f}<=auth<{high_t:.2f}<=decline | cost=${best_cost:,.0f}")
            print(f"  missed fraud:     n={missed_fraud_mask.sum():4d}   ${amt[missed_fraud_mask].sum():>10,.2f}")
            print(f"  fraud->auth:      n={fraud_to_auth_mask.sum():4d}   ${amt[fraud_to_auth_mask].sum():>10,.2f}  (of which ~${(1-CATCH_RATE)*amt[fraud_to_auth_mask].sum():,.2f} expected to still slip through)")
            print(f"  fraud declined:   n={fraud_declined_mask.sum():4d}   ${amt[fraud_declined_mask].sum():>10,.2f}  (blocked)")
            print(f"  legit declined:   n={legit_declined_mask.sum():4d}   ${amt[legit_declined_mask].sum():>10,.2f}  (false alarms, revenue lost outright)")
            print(f"  legit->auth:      n={legit_to_auth_mask.sum():4d}   ${amt[legit_to_auth_mask].sum():>10,.2f}  (false alarms, friction only)")
        print()

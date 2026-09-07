"""Sweep candidate velocity window sizes and measure each one's standalone
discriminative power against the real is_fraud label (ROC-AUC of that single
feature, before it's combined with anything else) — lets us pick window sizes
by evidence instead of guessing 1h/24h.

    python research/tune_velocity_windows.py
"""
import _bootstrap as bootstrap
import utils.imports as imports
import utils.features as features
from sklearn.metrics import roc_auc_score

CANDIDATE_WINDOWS = ['30min', '1h', '3h', '6h', '12h', '24h', '48h', '72h', '7d']

if __name__ == '__main__':
    config = imports.import_yml(bootstrap.CONFIG_PATH)
    cfg = config['fraud_test']
    feat_cfg = cfg['features']
    vel_cfg = feat_cfg['velocity']

    raw_df = imports.import_csv(cfg['data']['path'])
    raw_df[feat_cfg['time_col']] = raw_df[feat_cfg['time_col']].astype('datetime64[ns]')
    target_col = cfg['target_col']

    results = []
    for window in CANDIDATE_WINDOWS:
        df = raw_df.copy()
        df, count_col = features.rolling_count_velocity(df, vel_cfg['entity_col'], feat_cfg['time_col'], window)
        df, amt_col = features.rolling_amount_velocity(df, vel_cfg['entity_col'], feat_cfg['time_col'], vel_cfg['amount_col'], window)

        count_auc = roc_auc_score(df[target_col], df[count_col])
        amt_auc = roc_auc_score(df[target_col], df[amt_col])
        results.append((window, count_auc, amt_auc))
        print(f"{window:>7}   count_auc={count_auc:.4f}   amt_sum_auc={amt_auc:.4f}")

    print("\nsorted by count_auc:")
    for window, count_auc, amt_auc in sorted(results, key=lambda r: -r[1]):
        print(f"{window:>7}   count_auc={count_auc:.4f}   amt_sum_auc={amt_auc:.4f}")

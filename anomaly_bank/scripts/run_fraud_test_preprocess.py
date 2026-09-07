"""Run just the shared fraudTest.csv feature engineering and inspect the result.

    python scripts/run_preprocess.py
"""
import _bootstrap as bootstrap
import steps.fraud_test_preprocess as fraud_test_preprocess

if __name__ == '__main__':
    pre = fraud_test_preprocess.FraudTestPreprocess(config_path=bootstrap.CONFIG_PATH)
    raw_df, enriched_df, feat_columns = pre._execute()

    print(f"\nraw_df shape:      {raw_df.shape}")
    print(f"enriched_df shape: {enriched_df.shape}")
    print(f"feat_columns ({len(feat_columns)}): {feat_columns}")
    print("\nNaNs in feat_columns:")
    na_counts = enriched_df[feat_columns].isna().sum()
    print(na_counts[na_counts > 0] if na_counts.any() else "none")
    print("\nsample rows:")
    print(enriched_df[feat_columns + [pre.cfg['target_col']]].head())

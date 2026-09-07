"""Basic data-quality sanity checks on the raw fraudTest.csv, the same kind of
check that surfaced the DeviceID/IP artifacts on the old bank dataset — run once
rather than assuming this dataset is clean just because it has a real label.

    python scripts/data_quality_check.py
"""
import pandas as pd

import _bootstrap as bootstrap
import utils.imports as imports

if __name__ == '__main__':
    config = imports.import_yml(bootstrap.CONFIG_PATH)
    cfg = config['fraud_test']
    df = imports.import_csv(cfg['data']['path'])

    print(f"rows: {len(df)}")

    print(f"\nexact duplicate rows: {df.duplicated().sum()}")
    print(f"duplicate trans_num (should be a unique id): {df['trans_num'].duplicated().sum()}")

    print("\nmissing values per column:")
    na = df.isna().sum()
    print(na[na > 0] if na.any() else "none")

    print(f"\namt <= 0: {(df['amt'] <= 0).sum()}")
    print(f"amt range: [{df['amt'].min()}, {df['amt'].max()}]")

    for col in ['lat', 'long', 'merch_lat', 'merch_long']:
        print(f"{col} range: [{df[col].min()}, {df[col].max()}]")
    bad_lat = ((df['lat'].abs() > 90) | (df['merch_lat'].abs() > 90)).sum()
    bad_long = ((df['long'].abs() > 180) | (df['merch_long'].abs() > 180)).sum()
    print(f"out-of-range latitudes: {bad_lat}, longitudes: {bad_long}")

    dob = pd.to_datetime(df['dob'])
    tdt = pd.to_datetime(df['trans_date_trans_time'])
    print(f"\ndob range: [{dob.min()}, {dob.max()}]")
    print(f"trans_date_trans_time range: [{tdt.min()}, {tdt.max()}]")
    neg_age = (tdt < dob).sum()
    print(f"rows where transaction happens before dob (impossible age): {neg_age}")

    print(f"\ncc_num with only 1 transaction total: {(df['cc_num'].value_counts() == 1).sum()} of {df['cc_num'].nunique()} cards")

    print(f"\nis_fraud value counts:\n{df['is_fraud'].value_counts()}")

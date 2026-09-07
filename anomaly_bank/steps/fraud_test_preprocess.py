import pandas as pd

import utils.imports as imports
import utils.features as features


class FraudTestPreprocess():
    """Shared feature engineering for fraudTest.csv, consumed by both the Isolation
    Forest and XGBoost tracks so they train on identical features."""

    def __init__(self, config_path: str):
        self.config = imports.import_yml(config_path)
        self.cfg = self.config['fraud_test']
        self.raw_df = None
        self.enriched_df = None
        self.feat_columns = []

    def import_raw_data(self):
        self.raw_df = imports.import_csv(self.cfg['data']['path'])
        return None

    def preprocess_data(self):
        feat_cfg = self.cfg['features']
        self.raw_df[feat_cfg['time_col']] = self.raw_df[feat_cfg['time_col']].astype('datetime64[ns]')
        self.raw_df[feat_cfg['dob_col']] = self.raw_df[feat_cfg['dob_col']].astype('datetime64[ns]')
        self.raw_df['age'] = ((self.raw_df[feat_cfg['time_col']] - self.raw_df[feat_cfg['dob_col']]).dt.days // 365).astype(int)
        self.feat_columns.append('age')

        self.raw_df['hour_of_day'] = self.raw_df[feat_cfg['time_col']].dt.hour
        self.raw_df['day_of_week'] = self.raw_df[feat_cfg['time_col']].dt.dayofweek
        self.feat_columns.extend(['hour_of_day', 'day_of_week'])
        return None

    def engineer_geo_features(self):
        geo_cfg = self.cfg['features']['geo']
        self.enriched_df, dist_col = features.haversine_distance_column(
            self.enriched_df, geo_cfg['customer_lat'], geo_cfg['customer_long'],
            geo_cfg['merchant_lat'], geo_cfg['merchant_long'],
        )
        self.feat_columns.append(dist_col)

    def engineer_velocity_features(self):
        """Same-card/same-merchant rolling count+amount velocity (rolling_count_velocity/
        rolling_amount_velocity in utils/features.py) was tested here — 9 window sizes
        (30min-7d), raw and per-card-relative framings, both cc_num and merchant entities —
        and found to carry no signal in this dataset (flat ~0.50 AUC throughout, confirmed
        weakest features by SHAP). Not computed here for now. The functions are kept as-is
        for Phase 2, where the same rolling-window mechanism applies to genuinely different
        signals (device/network-level velocity) that may behave differently.
        See scripts/tune_velocity_windows.py for the sweep.
        """
        vel_cfg = self.cfg['features']['velocity']
        time_col = self.cfg['features']['time_col']

        # how unusual amt is vs. this card's own prior history (no lookahead) — this one showed real signal
        self.enriched_df, zscore_col = features.expanding_zscore(
            self.enriched_df, vel_cfg['entity_col'], time_col, vel_cfg['amount_col']
        )
        self.feat_columns.append(zscore_col)

    def engineer_categorical_features(self):
        feat_cfg = self.cfg['features']

        for column_name in feat_cfg['target_encoding_columns']:
            self.enriched_df = features.target_encoding_kfold(
                self.enriched_df, column_name, self.cfg['target_col']
            )
            self.feat_columns.append(f"{column_name}_te")

        for column_name in feat_cfg.get('freq_encoding_columns', []):
            self.enriched_df, freq_column = features.frequency_encoding_column(self.enriched_df, column_name)
            self.feat_columns.append(freq_column)

        for column_name in feat_cfg['one_hot_encoding_columns']:
            self.enriched_df, one_hot_columns = features.one_hot_encoding_column(self.enriched_df, column_name)
            self.feat_columns.extend(one_hot_columns)

    def engineer_features(self):
        self.enriched_df = self.raw_df.copy()
        self.engineer_geo_features()
        self.engineer_velocity_features()
        self.engineer_categorical_features()
        self.feat_columns.extend(self.cfg['features']['numerical_columns'])
        return None

    def _execute(self):
        self.import_raw_data()
        self.preprocess_data()
        self.engineer_features()
        return self.raw_df, self.enriched_df, self.feat_columns

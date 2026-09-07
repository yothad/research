"""Isolation Forest for bank_transactions_data.csv, which has no real fraud label.
For fraudTest.csv (which does have a real label), see fraud_test_isolation_forest.py's
FraudTestIsolationForestModel — much leaner, since it can evaluate directly against
ground truth instead of needing the SHAP-ranked feature sweep / explainability /
confidence machinery this file builds to make an unsupervised model usable without one.
"""
import sys
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', 500)

import utils.imports as imports
import utils.features as features
import utils.models as models


class BankIsolationForestPreprocess():
    def __init__(self, config_path: str):
        self.config = imports.import_yml(config_path)
        self.raw_df = None
        self.enriched_df = None
        self.feat_columns = ['TransactionAmount',
                             'TransactionDuration',
                             'LoginAttempts',
                             'AccountBalance',
                             'CustomerAge']

    def import_raw_data(self):
        print(self.config['data']['path'])
        print(Path().resolve())
        self.raw_df = imports.import_csv(self.config['data']['path'])
        return None

    def preprocess_data(self):
        # change dtypes
        for date_col in self.config['isolation_forest']['features']['datetime_columns']:
            self.raw_df[date_col] = self.raw_df[date_col].astype('datetime64[ns]')

        # rename columns
        self.raw_df = self.raw_df.rename(columns={'IP Address': 'IP_Address',
                                                  'TransactionDate': 'PreviousTransactionDate',
                                                  'PreviousTransactionDate': 'TransactionDate',
                                                  })

        # Extract IP prefix
        self.raw_df = features.extract_ip_prefix(self.raw_df, 'IP_Address', num_levels=self.config['isolation_forest']['features']['IP_Address_extract_level'])
        self.raw_df['diff_days'] = (self.raw_df['TransactionDate'] - self.raw_df['PreviousTransactionDate']).dt.days
        self.feat_columns.append('diff_days')

        return None

    def engineer_features(self):
        self.enriched_df = self.raw_df.copy()

        for column_name in self.config['isolation_forest']['features']['freq_encoding_columns']:
            self.enriched_df, freq_column = features.frequency_encoding_column(self.enriched_df, column_name)
            self.feat_columns.append(freq_column)

        for column_name in self.config['isolation_forest']['features']['one_hot_encoding_columns']:
            self.enriched_df, one_hot_columns = features.one_hot_encoding_column(self.enriched_df, column_name)
            self.feat_columns.extend(one_hot_columns)

    def aggregate_features(self):
        gb_df = self.enriched_df.groupby('AccountID', as_index=False).agg(TransactionAmount_sum=('TransactionAmount', 'sum'),
                                                                          TransactionAmount_mean=('TransactionAmount', 'mean'),
                                                                          TransactionID_count=('TransactionID', 'count'),
                                                                          IP_nunique=('IP_Address', 'nunique'),
                                                                          Merchant_nunique=('MerchantID', 'nunique'),
                                                                          )
        self.enriched_df = self.enriched_df.merge(gb_df, how='inner', on='AccountID')
        self.feat_columns.extend(['TransactionAmount_sum',
                                  'TransactionAmount_mean',
                                  'TransactionID_count',
                                  'IP_nunique',
                                  'Merchant_nunique'])
        return None

    def _execute(self):
        self.import_raw_data()
        self.preprocess_data()
        self.engineer_features()
        self.aggregate_features()
        self.feat_columns
        return self.raw_df, self.enriched_df, self.feat_columns


class BankIsolationForestModel():
    def __init__(self,
                 config_path: str,
                 raw_df: pd.DataFrame,
                 enriched_df: pd.DataFrame,
                 feat_columns: list):
        self.config = imports.import_yml(config_path)
        self.raw_df = raw_df
        self.enriched_df = enriched_df
        self.all_feat_columns = feat_columns

    def run_model(self, feat_columns: list):
        isolation_model, isolation_df, data_arr = models.run_isolation_forest(df = self.enriched_df,
                                                                              features = feat_columns,
                                                                              n_estimators = self.config['isolation_forest']['model']['n_estimators'],
                                                                              contamination = self.config['isolation_forest']['model']['contamination'],
                                                                              random_state = self.config['isolation_forest']['model']['random_state'],
                                                                              max_samples = self.config['isolation_forest']['model']['max_samples'],
                                                                              )
        return isolation_model, isolation_df, data_arr

    def compute_shap(self, model, data_arr):
        importance_df = models.compute_shap_values(model, data_arr, self.all_feat_columns)
        return importance_df

    def compute_row_explanations(self, model, data_arr, feat_columns, top_n: int = 3, min_abs_shap: float = 1.0):
        """Per-transaction SHAP breakdown: which features pushed each row toward/away from anomalous."""
        shap_explanation = models.compute_shap_explanation(model, data_arr, feat_columns)
        top_reasons = models.summarize_shap_reasons(shap_explanation, top_n=top_n, min_abs_shap=min_abs_shap)
        return shap_explanation, top_reasons

    def compute_confidence(self, model, data_arr):
        """Tree-vote agreement: how consistently the ensemble's trees isolate each row."""
        return models.compute_tree_agreement(model, data_arr)

    def compute_within_anomaly_confidence(self, isolation_df: pd.DataFrame) -> pd.Series:
        """Confidence percentile ranked only among flagged anomalies, so it differentiates
        within that group instead of being dominated by the anomaly/normal split."""
        mask = isolation_df['Anomaly'] == -1
        within_confidence = pd.Series(index=isolation_df.index, dtype=float)
        within_confidence.loc[mask] = (-isolation_df.loc[mask, 'TreeDepthStd']).rank(pct=True) * 100
        return within_confidence

    def compute_anomaly_stats(self, isolation_df: pd.DataFrame,
                              feat_columns: list,
                              median_weight: float = 0.4,
                              mean_weight: float = 0.4,
                              stdev_weight: float = -0.2,
                              ):
        anomaly_agg_df =  pd.DataFrame({
            'anomaly_score_median': [isolation_df[isolation_df['Anomaly'] == -1]['AnomalyScore'].median()],
            'anomaly_score_mean': [isolation_df[isolation_df['Anomaly'] == -1]['AnomalyScore'].mean()],
            'anomaly_score_stdev': [isolation_df[isolation_df['Anomaly'] == -1]['AnomalyScore'].std()],
            'feat_columns': [feat_columns],
            'feat_columns_count': [len(feat_columns)],
        })
        anomaly_agg_df['anomaly_final_score'] = median_weight*anomaly_agg_df['anomaly_score_median'] +mean_weight*anomaly_agg_df['anomaly_score_median'] + stdev_weight*anomaly_agg_df['anomaly_score_median']
        return anomaly_agg_df

    def iterate_features_in_model(self):
        # run initial model to get feature importances
        isolation_model, isolation_df, data_arr = self.run_model(feat_columns=self.all_feat_columns)
        importance_df = self.compute_shap(model=isolation_model, data_arr=data_arr)
        anomaly_aggs_df = self.compute_anomaly_stats(isolation_df, self.all_feat_columns)

        # loop through number of features by importance
        print(f"Iterating through {len(importance_df)} feature counts to find best model...")
        for i in range(0, len(importance_df)-1):
            selected_features = importance_df.iloc[0:i+1]['feature'].to_list()
            # print(f"Running model with top {i} features, features: {selected_features}")
            isolation_model, isolation_df, data_arr = self.run_model(feat_columns=selected_features)
            anomaly_aggs_df_iter = self.compute_anomaly_stats(isolation_df, selected_features)
            anomaly_aggs_df = pd.concat([anomaly_aggs_df, anomaly_aggs_df_iter], axis=0).reset_index(drop=True)

        return anomaly_aggs_df

    def select_best_features(self, anomaly_aggs_df: pd.DataFrame):
        best_row = anomaly_aggs_df.loc[anomaly_aggs_df['anomaly_final_score'].idxmax()]
        best_features = best_row['feat_columns']
        print(f"Model features with best params: {best_row}")
        return best_features

    def post_process_results(self, isolation_df: pd.DataFrame):
        columns = self.raw_df.columns.tolist()
        extra_columns = ['Anomaly', 'AnomalyScore', 'TopReasons', 'Confidence', 'TreeDepthStd', 'ConfidenceWithinAnomalies']
        rename_map = {
            'Anomaly': 'IsolationForest_Anomaly',
            'AnomalyScore': 'IsolationForest_AnomalyScore',
            'TopReasons': 'IsolationForest_TopReasons',
            'Confidence': 'IsolationForest_Confidence',
            'TreeDepthStd': 'IsolationForest_TreeDepthStd',
            'ConfidenceWithinAnomalies': 'IsolationForest_ConfidenceWithinAnomalies',
        }
        labeled_raw_df = self.raw_df.merge(isolation_df[columns + extra_columns].rename(columns=rename_map), how='left', on=columns)
        labeled_raw_df.loc[labeled_raw_df['IsolationForest_Anomaly'] == 1, 'IsolationForest_Anomaly'] = 0
        labeled_raw_df.loc[labeled_raw_df['IsolationForest_Anomaly'] == -1, 'IsolationForest_Anomaly'] = 1
        return labeled_raw_df

    def _execute(self):
        anomaly_aggs_df = self.iterate_features_in_model()
        best_features = self.select_best_features(anomaly_aggs_df)
        isolation_model, isolation_df, data_arr = self.run_model(feat_columns=best_features)
        shap_explanation, top_reasons = self.compute_row_explanations(isolation_model, data_arr, best_features)
        confidence_df = self.compute_confidence(isolation_model, data_arr)
        isolation_df['TopReasons'] = top_reasons.values
        isolation_df['Confidence'] = confidence_df['Confidence'].values
        isolation_df['TreeDepthStd'] = confidence_df['TreeDepthStd'].values
        isolation_df['ConfidenceWithinAnomalies'] = self.compute_within_anomaly_confidence(isolation_df)
        labeled_raw_df = self.post_process_results(isolation_df)

        # exposed so a fitted model can be reused for scoring new data (e.g. an eval harness) without refitting
        self.model = isolation_model
        self.best_features = best_features

        return anomaly_aggs_df, isolation_df, labeled_raw_df, shap_explanation

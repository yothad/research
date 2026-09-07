import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

import utils.imports as imports
import utils.models as models


class FraudTestXGBoostModel():
    """Supervised track for fraudTest.csv: trains directly on the real is_fraud label.
    Phase 1 scope: stratified split, class-imbalance-aware fit, baseline metrics.
    Risk-score calibration/banding, the rules layer, outcome routing, and the
    cost-based threshold are later phases built on top of this.
    """

    def __init__(self, config_path: str, enriched_df: pd.DataFrame, feat_columns: list):
        self.config = imports.import_yml(config_path)
        self.cfg = self.config['fraud_test']['xgboost']
        self.enriched_df = enriched_df
        self.feat_columns = feat_columns
        self.model = None

    def split(self, target_col: str):
        df = self.enriched_df[self.feat_columns + [target_col]]
        return models.split_data(df, target_col=target_col, test_size=self.cfg['test_size'], random_state=self.cfg['random_state'])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        self.model = xgb.XGBClassifier(
            n_estimators=self.cfg['n_estimators'],
            learning_rate=self.cfg['learning_rate'],
            max_depth=self.cfg['max_depth'],
            random_state=self.cfg['random_state'],
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
        )
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_score = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        return {
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_test, y_score),
        }

    def _execute(self, target_col: str):
        X_train, X_test, y_train, y_test = self.split(target_col)
        self.fit(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        return metrics

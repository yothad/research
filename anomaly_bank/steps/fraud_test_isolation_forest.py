import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

import utils.imports as imports
import utils.models as models


class FraudTestIsolationForestModel():
    """Lean unsupervised track for fraudTest.csv: fit once on the shared feature set
    (no is_fraud in the features), then compare directly against the real label.
    No pseudo-labeling / SHAP-ranked feature sweep needed here — that machinery in
    steps/bank_isolation_forest.py's BankIsolationForestModel exists because the bank
    dataset had no ground truth; this dataset has one, so evaluation is direct. Both
    classes call the same underlying utils/models.py::run_isolation_forest — this one
    just skips the explainability/confidence layer that dataset needed and this one doesn't.
    """

    def __init__(self, config_path: str, enriched_df: pd.DataFrame, feat_columns: list):
        self.config = imports.import_yml(config_path)
        self.cfg = self.config['fraud_test']['isolation_forest']
        self.enriched_df = enriched_df
        self.feat_columns = feat_columns
        self.model = None

    def run_model(self) -> pd.DataFrame:
        self.model, isolation_df, data_arr = models.run_isolation_forest(
            df=self.enriched_df,
            features=self.feat_columns,
            n_estimators=self.cfg['n_estimators'],
            contamination=self.cfg['contamination'],
            random_state=self.cfg['random_state'],
            max_samples=self.cfg['max_samples'],
        )
        return isolation_df

    def evaluate_against_label(self, isolation_df: pd.DataFrame, target_col: str) -> dict:
        y_true = isolation_df[target_col]
        y_pred = (isolation_df['Anomaly'] == -1).astype(int)
        y_score = isolation_df['AnomalyScore']

        return {
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_true, y_score),
            'n_flagged': int(y_pred.sum()),
            'n_actual_fraud': int(y_true.sum()),
        }

    def _execute(self, target_col: str):
        isolation_df = self.run_model()
        metrics = self.evaluate_against_label(isolation_df, target_col)
        return isolation_df, metrics

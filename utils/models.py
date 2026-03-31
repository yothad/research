import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score  # Optional if labels exist
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def run_isolation_forest(df: pd.DataFrame, 
                         features: list,
                         n_estimators: int = 100,
                         contamination: float = 0.01,
                         random_state: int = 42,
                         max_samples: str = "auto"):
    data_arr = df[features].fillna(0)

    # Fit
    model = IsolationForest(n_estimators=n_estimators, 
                            contamination=contamination,
                            max_samples=max_samples,
                            random_state=random_state,
                            )
    model.fit(data_arr)

    # Predict
    df['Anomaly'] = model.predict(data_arr)            # -1 = anomaly, 1 = normal
    df['AnomalyScore'] = -model.decision_function(data_arr)  # Higher = more anomalous

    return model, df, data_arr

def compute_shap_values(model, data_arr, features):
    explainer = shap.Explainer(model, data_arr)
    shap_values = explainer(data_arr)

    # Mean absolute SHAP values for global feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_shap_val': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='mean_shap_val', ascending=False).reset_index(drop=True)

    return importance_df


def split_data(df, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        features = [col for col in df.columns if col != "churn"]
        X = df[features]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    

def smote_resample(X_train, y_train, random_state=42):
    """Apply SMOTE to balance the training data."""
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm

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

def score_isolation_forest(model, df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Score new data with an already-fitted IsolationForest, without refitting."""
    df = df.copy()
    data_arr = df[features].fillna(0)
    df['Anomaly'] = model.predict(data_arr)
    df['AnomalyScore'] = -model.decision_function(data_arr)
    return df


def compute_shap_values(model, data_arr, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data_arr)

    # Mean absolute SHAP values for global feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_shap_val': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='mean_shap_val', ascending=False).reset_index(drop=True)

    return importance_df


def compute_shap_explanation(model, data_arr, features):
    """Per-row SHAP values (not aggregated) for the given model, used to explain individual predictions."""
    explainer = shap.TreeExplainer(model)
    return explainer(data_arr)


def summarize_shap_reasons(shap_explanation, top_n: int = 3, min_abs_shap: float = 1.0) -> pd.Series:
    """For each row, up to top_n features driving its score: name, actual value, and signed SHAP contribution.

    Only features whose |SHAP| clears min_abs_shap are included, so weak/incidental
    contributors (e.g. a one-hot flag with a small nudge) don't clutter genuinely strong reasons.
    Rows with no feature clearing the bar get a fallback message.
    """
    features = shap_explanation.feature_names
    shap_df = pd.DataFrame(shap_explanation.values, columns=features)
    data_df = pd.DataFrame(shap_explanation.data, columns=features)

    def format_val(v: float) -> str:
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"

    def top_reasons(idx):
        row = shap_df.loc[idx]
        ranked = row.reindex(row.abs().sort_values(ascending=False).index)
        ranked = ranked[ranked.abs() >= min_abs_shap].head(top_n)
        if len(ranked) == 0:
            return "no single dominant driver (all contributions below threshold)"
        return "; ".join(f"{feat}={format_val(data_df.loc[idx, feat])} ({val:+.3f})" for feat, val in ranked.items())

    return pd.Series([top_reasons(idx) for idx in shap_df.index], index=shap_df.index)


def split_data(df, target_col: str, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        features = [col for col in df.columns if col != target_col]
        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    

def compute_tree_agreement(model, data_arr) -> pd.DataFrame:
    """Per-row confidence signal from how much the ensemble's individual trees agree.

    Each tree in an IsolationForest isolates a row after some number of splits (its depth).
    Averaging depth across trees gives the score; the *spread* across trees is a separate
    signal the score alone doesn't carry: low std means every tree isolated this row similarly
    (confident), high std means trees disagree (a borderline call, often driven by a feature
    that only some trees happened to split on).
    """
    X = data_arr.values if hasattr(data_arr, 'values') else np.asarray(data_arr)
    depths = np.vstack([
        np.asarray(est.decision_path(X).sum(axis=1)).ravel() - 1
        for est in model.estimators_
    ])  # shape: (n_estimators, n_samples)

    tree_depth_std = depths.std(axis=0)
    confidence = pd.Series(-tree_depth_std).rank(pct=True) * 100  # 0-100, higher = more tree agreement

    return pd.DataFrame({
        'TreeDepthMean': depths.mean(axis=0),
        'TreeDepthStd': tree_depth_std,
        'Confidence': confidence.values,
    })


def smote_resample(X_train, y_train, random_state=42):
    """Apply SMOTE to balance the training data."""
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm

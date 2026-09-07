"""3-tier routing policy: Auto Pass / Human Authentication / Auto Decline,
using cost-optimal thresholds on the raw XGBoost fraud score. The thresholds
(config.yml's fraud_test.routing) came from an elbow search over a log-scaled
false-decline cost model — full reasoning trail in
research/cost_model_findings.md, methodology script in
research/three_tier_routing_analysis.py.
"""
import numpy as np
import pandas as pd

import utils.imports as imports

AUTO_PASS = 'auto_pass'
HUMAN_AUTH = 'human_auth'
AUTO_DECLINE = 'auto_decline'


class RiskRouter():
    def __init__(self, config_path: str):
        self.config = imports.import_yml(config_path)
        self.cfg = self.config['fraud_test']['routing']

    def route(self, score) -> pd.Series:
        """score: array-like of raw fraud-probability scores (e.g. from
        XGBClassifier.predict_proba(...)[:, 1]). Returns one decision per row:
        AUTO_PASS, HUMAN_AUTH, or AUTO_DECLINE."""
        score = np.asarray(score)
        decisions = np.full(score.shape, HUMAN_AUTH, dtype=object)
        decisions[score < self.cfg['pass_threshold']] = AUTO_PASS
        decisions[score >= self.cfg['decline_threshold']] = AUTO_DECLINE
        return pd.Series(decisions, name='routing_decision')

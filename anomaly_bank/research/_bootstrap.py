"""Shared path setup for runnable scripts (scripts/, research/, or any future
sibling folder under anomaly_bank/). Import this first, before `steps.*`/`utils.*`,
so both packages resolve and relative paths in config.yml (e.g. data/fraudTest.csv)
work regardless of where the script is invoked from. Each such folder needs its own
copy of this exact file — Python only auto-adds the *running script's own* directory
to sys.path, not sibling folders, so a bare `import _bootstrap` only resolves locally.
"""
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve()
ANOMALY_BANK_ROOT = next(
    p for p in [_here.parent] + list(_here.parents) if (p / 'config' / 'config.yml').exists()
)
REPO_ROOT = ANOMALY_BANK_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ANOMALY_BANK_ROOT))
os.chdir(ANOMALY_BANK_ROOT)

CONFIG_PATH = "config/config.yml"

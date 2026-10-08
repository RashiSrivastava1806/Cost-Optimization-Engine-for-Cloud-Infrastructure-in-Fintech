from pathlib import Path

import numpy as np
import pandas as pd

# 1. PATHS AND CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_FILE = DATA_DIR / "engineered_test_cloud_billing_data.csv"

TRAIN_OUTPUT = DATA_DIR / "anomaly_train_cloud_billing_data.csv"
TEST_OUTPUT = DATA_DIR / "anomaly_test_cloud_billing_data.csv"
SUMMARY_OUTPUT = DATA_DIR / "anomaly_detection_summary.csv"
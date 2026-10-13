"""
SLA Compliance Configuration
Purpose:
-Centralize SLA screening thresholds and policy settings.
-Separate configurable engineering guardrails from contractual SLA limits.
-Define consistent input/output paths and decision labels.

Important:
The current dataset does not contain contractual SLA targets, uptime,latency,or transaction-performance measurements.Therefore,these thresholds support preliminary risk screening , not SLA certification .
"""
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input files produced by the Anomaly Detection stage
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_INPUT_FILE = DATA_DIR / "anomaly_train_cloud_billing_data.csv"
TEST_INPUT_FILE = DATA_DIR / "anomaly_test_cloud_billing_data.csv"

# Output files produced by the SLA Compliance stage
TRAIN_OUTPUT_FILE = DATA_DIR / "sla_train_cloud_billing_data.csv"
TEST_OUTPUT_FILE = DATA_DIR / "sla_test_cloud_billing_data.csv"

SUMMARY_OUTPUT_FILE = DATA_DIR / "sla_compliance_summary.csv"


# ============================================================
# 2. UTILIZATION THRESHOLDS
# ============================================================

# CPU utilization thresholds (%)
CPU_BLOCK_THRESHOLD = 85.0
CPU_REVIEW_THRESHOLD = 70.0
CPU_LOW_UTILIZATION_THRESHOLD = 30.0

# Memory utilization thresholds (%)
MEMORY_BLOCK_THRESHOLD = 85.0
MEMORY_REVIEW_THRESHOLD = 75.0
MEMORY_LOW_UTILIZATION_THRESHOLD = 40.0

# Interpretation:
# - At or above a block threshold: block downsizing.
# - At or above a review threshold but below the block threshold:
#   require manual review.
# - Low CPU AND low memory: potential optimization candidate.
#
# These are provisional engineering guardrails, not contractual
# SLA limits. Actual service-specific policies must override them.


# ============================================================
# 3. COST CONTEXT SETTINGS
# ============================================================

# Existing contextual cost signal used in anomaly detection.
# A resource costing at least 1.25 times its service benchmark may
# warrant investigation, but this does not independently prove waste.
HIGH_COST_RATIO_THRESHOLD = 1.25

# Do not treat cost alone as evidence of an SLA violation.
COST_SIGNAL_REQUIRES_REVIEW = True


# ============================================================
# 4. DATA QUALITY SETTINGS
# ============================================================

# Expected fields available in the current billing dataset.
REQUIRED_COLUMNS = (
    "Resource_ID",
    "Service_Name",
    "CPU_Utilization",
    "Memory_Utilization",
    "Usage_Start_Date",
    "Usage_End_Date",
    "Total_Cost_INR",
)

# Evidence needed for genuine SLA verification.
# These fields are not currently available in the supplied dataset.
SLA_EVIDENCE_COLUMNS = (
    "SLA_Target",
    "Criticality",
    "Availability_Percent",
    "Response_Time_ms",
)

# Missing SLA evidence must not be interpreted as compliance.
MISSING_SLA_EVIDENCE_STATUS = "NOT_VERIFIABLE"


# ============================================================
# 5. SLA COMPLIANCE LABELS
# ============================================================

SLA_STATUS_COMPLIANT = "COMPLIANT"
SLA_STATUS_VIOLATION = "VIOLATION_DETECTED"
SLA_STATUS_NOT_VERIFIABLE = "NOT_VERIFIABLE"

SLA_STATUS_VALUES = (
    SLA_STATUS_COMPLIANT,
    SLA_STATUS_VIOLATION,
    SLA_STATUS_NOT_VERIFIABLE,
)


# ============================================================
# 6. RESOURCE RISK LEVELS
# ============================================================

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"
RISK_UNKNOWN = "UNKNOWN"

RISK_LEVEL_VALUES = (
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_LOW,
    RISK_UNKNOWN,
)


# ============================================================
# 7. OPTIMIZATION DECISIONS
# ============================================================

DECISION_BLOCKED = "BLOCKED"
DECISION_REVIEW_REQUIRED = "REVIEW_REQUIRED"
DECISION_PROVISIONAL_CANDIDATE = "PROVISIONAL_CANDIDATE"

OPTIMIZATION_DECISION_VALUES = (
    DECISION_BLOCKED,
    DECISION_REVIEW_REQUIRED,
    DECISION_PROVISIONAL_CANDIDATE,
)


# ============================================================
# 8. ANOMALY HANDLING
# ============================================================

# Existing anomaly flags should trigger investigation before a
# resource is considered for optimization.
ANOMALY_REQUIRES_INVESTIGATION = True

# An anomaly is not automatically an SLA violation.
ANOMALY_IMPLIES_SLA_VIOLATION = False


# ============================================================
# 9. OPTIMIZATION SAFETY SETTINGS
# ============================================================

# This project generates recommendations; it does not change
# cloud resources automatically.
AUTOMATIC_RESOURCE_MODIFICATION_ENABLED = False

# A provisional candidate is not final approval.
REQUIRE_MANUAL_APPROVAL = True

# Missing SLA evidence prevents definitive SLA certification.
REQUIRE_SLA_EVIDENCE_FOR_CERTIFICATION = True


# ============================================================
# 10. REPORTING SETTINGS
# ============================================================

SUMMARY_GROUP_COLUMNS = (
    "SLA_Compliance_Status",
    "SLA_Risk_Level",
    "Optimization_Decision",
)

# Output fields expected from later SLA pipeline components.
EXPECTED_OUTPUT_COLUMNS = (
    "SLA_Compliance_Status",
    "SLA_Risk_Level",
    "Optimization_Decision",
    "SLA_Data_Complete",
    "SLA_Checks_Performed",
    "Optimization_Block_Reason",
    "SLA_Risk_Reason",
)

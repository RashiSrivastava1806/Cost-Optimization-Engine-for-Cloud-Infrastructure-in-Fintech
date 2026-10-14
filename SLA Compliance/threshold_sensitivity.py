
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "canonical_run"
FIGURES = ROOT / "Research Results" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# These are experimental configurations, not industry standards.
CONFIGS = {
    "Conservative": {
        "cpu_review": 60,
        "cpu_block": 75,
        "memory_review": 65,
        "memory_block": 75,
    },
    "Proposed": {
        "cpu_review": 70,
        "cpu_block": 85,
        "memory_review": 75,
        "memory_block": 85,
    },
    "Relaxed": {
        "cpu_review": 80,
        "cpu_block": 90,
        "memory_review": 80,
        "memory_block": 90,
    },
}


def classify(df, config):
    cpu = pd.to_numeric(df["Evaluation_CPU"], errors="coerce")
    memory = pd.to_numeric(df["Evaluation_Memory"], errors="coerce")
    anomaly = df["Detected_Anomaly"].fillna(False).astype(bool)

    valid = (
        cpu.notna()
        & memory.notna()
        & cpu.between(0, 100)
        & memory.between(0, 100)
    )

    invalid = (
        (cpu.notna() & ~cpu.between(0, 100))
        | (memory.notna() & ~memory.between(0, 100))
    )

    blocked = (
        (cpu >= config["cpu_block"])
        | (memory >= config["memory_block"])
    ).fillna(False)

    review = (
        (
            (cpu >= config["cpu_review"])
            & (cpu < config["cpu_block"])
        )
        | (
            (memory >= config["memory_review"])
            & (memory < config["memory_block"])
        )
    ).fillna(False)

    low_utilization = (
        (cpu <= 30)
        & (memory <= 40)
    ).fillna(False)

    # Keep decision priority consistent with the main framework.
    decision = pd.Series(
        "ROUTINE_MONITORING", index=df.index, dtype="object"
    )

    decision.loc[low_utilization & ~anomaly] = (
        "PROVISIONAL_CANDIDATE"
    )
    decision.loc[anomaly] = "REVIEW_REQUIRED"
    decision.loc[review] = "REVIEW_REQUIRED"
    decision.loc[~valid] = "REVIEW_REQUIRED"
    decision.loc[invalid] = "REVIEW_REQUIRED"
    decision.loc[blocked] = "BLOCKED"

    return decision


records = []

for dataset_name in ["train", "test"]:
    path = DATA / f"fintech_sla_{dataset_name}_results.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run fintech_sla_framework.py first."
        )

    df = pd.read_csv(path)

    for config_name, config in CONFIGS.items():
        decisions = classify(df, config)

        records.append({
            "Dataset": dataset_name.upper(),
            "Configuration": config_name,
            "Total_Resources": len(df),
            "Blocked": int(
                (decisions == "BLOCKED").sum()
            ),
            "Review_Required": int(
                (decisions == "REVIEW_REQUIRED").sum()
            ),
            "Provisional_Candidates": int(
                (decisions == "PROVISIONAL_CANDIDATE").sum()
            ),
            "Routine_Monitoring": int(
                (decisions == "ROUTINE_MONITORING").sum()
            ),
            "CPU_Review_Threshold": config["cpu_review"],
            "CPU_Block_Threshold": config["cpu_block"],
            "Memory_Review_Threshold": config["memory_review"],
            "Memory_Block_Threshold": config["memory_block"],
        })


results = pd.DataFrame(records)

csv_path = DATA / "threshold_sensitivity_results.csv"
results.to_csv(csv_path, index=False)

print("\n=== THRESHOLD SENSITIVITY RESULTS ===")
print(
    results[
        [
            "Dataset",
            "Configuration",
            "Blocked",
            "Review_Required",
            "Provisional_Candidates",
            "Routine_Monitoring",
        ]
    ].to_string(index=False)
)

# Plot test-set results only.
test_results = results[results["Dataset"] == "TEST"].set_index(
    "Configuration"
).loc[list(CONFIGS.keys())]

ax = test_results[
    [
        "Blocked",
        "Review_Required",
        "Provisional_Candidates",
        "Routine_Monitoring",
    ]
].plot(kind="bar", figsize=(10, 6))

ax.set_title("Sensitivity of Resource Decisions to SLA Risk Thresholds")
ax.set_xlabel("Experimental threshold configuration")
ax.set_ylabel("Number of test resources")
ax.legend(title="Decision")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

figure_path = FIGURES / "threshold_sensitivity.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nCSV saved: {csv_path}")
print(f"Figure saved: {figure_path}")

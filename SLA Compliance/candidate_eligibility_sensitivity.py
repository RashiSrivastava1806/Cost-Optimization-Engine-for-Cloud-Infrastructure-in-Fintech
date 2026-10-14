
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "canonical_run"
FIGURES = ROOT / "Research Results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Hold proposed review/block thresholds fixed.
CPU_REVIEW = 70
CPU_BLOCK = 85
MEMORY_REVIEW = 75
MEMORY_BLOCK = 85

# Vary only the low-utilization eligibility thresholds.
CANDIDATE_CONFIGS = {
    "Strict": {"cpu_low": 20, "memory_low": 30},
    "Proposed": {"cpu_low": 30, "memory_low": 40},
    "Permissive": {"cpu_low": 40, "memory_low": 50},
}


def evaluate(df, thresholds):
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
        (cpu >= CPU_BLOCK) | (memory >= MEMORY_BLOCK)
    ).fillna(False)

    review = (
        (
            (cpu >= CPU_REVIEW)
            & (cpu < CPU_BLOCK)
        )
        | (
            (memory >= MEMORY_REVIEW)
            & (memory < MEMORY_BLOCK)
        )
    ).fillna(False)

    low_utilization = (
        (cpu <= thresholds["cpu_low"])
        & (memory <= thresholds["memory_low"])
    ).fillna(False)

    # Preserve the same decision priority as the main framework.
    decision = pd.Series(
        "ROUTINE_MONITORING", index=df.index, dtype="object"
    )
    decision.loc[low_utilization & ~anomaly] = "PROVISIONAL_CANDIDATE"
    decision.loc[anomaly] = "REVIEW_REQUIRED"
    decision.loc[review] = "REVIEW_REQUIRED"
    decision.loc[~valid | invalid] = "REVIEW_REQUIRED"
    decision.loc[blocked] = "BLOCKED"

    return decision


records = []

for dataset_name in ["TRAIN", "TEST"]:
    path = DATA / f"fintech_sla_{dataset_name.lower()}_results.csv"
    df = pd.read_csv(path)

    for config_name, thresholds in CANDIDATE_CONFIGS.items():
        decision = evaluate(df, thresholds)

        records.append({
            "Dataset": dataset_name,
            "Configuration": config_name,
            "CPU_Low_Threshold": thresholds["cpu_low"],
            "Memory_Low_Threshold": thresholds["memory_low"],
            "Total_Resources": len(df),
            "Provisional_Candidates": int(
                (decision == "PROVISIONAL_CANDIDATE").sum()
            ),
            "Blocked": int((decision == "BLOCKED").sum()),
            "Review_Required": int(
                (decision == "REVIEW_REQUIRED").sum()
            ),
            "Routine_Monitoring": int(
                (decision == "ROUTINE_MONITORING").sum()
            ),
        })

results = pd.DataFrame(records)
out = DATA / "candidate_eligibility_sensitivity_results.csv"
results.to_csv(out, index=False)

print("\n=== CANDIDATE ELIGIBILITY SENSITIVITY ===")
print(results.to_string(index=False))

test = results[results["Dataset"] == "TEST"].copy()
test = test.set_index("Configuration").loc[
    list(CANDIDATE_CONFIGS.keys())
]

ax = test["Provisional_Candidates"].plot(
    kind="bar", figsize=(8, 5)
)
ax.set_title("Candidate Eligibility Sensitivity on Held-Out Test Data")
ax.set_xlabel("Low-utilization threshold configuration")
ax.set_ylabel("Provisional candidate count")
ax.tick_params(axis="x", rotation=0)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

figure = FIGURES / "candidate_eligibility_sensitivity.png"
plt.savefig(figure, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nCSV saved: {out}")
print(f"Figure saved: {figure}")

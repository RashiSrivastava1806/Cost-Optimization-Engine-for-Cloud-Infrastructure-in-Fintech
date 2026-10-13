# SLA Compliance and Risk Assessment Module

## 1. Overview

This module is part of the Cost Optimization Engine for Cloud Infrastructure in FinTech. It evaluates cloud resource utilization, identifies potential optimization opportunities, assigns heuristic risk scores, and applies safety controls before resources are considered for optimization.

The system prioritizes cost optimization while avoiding decisions that could compromise service reliability or performance.

## 2. Objectives

- Validate the quality of input cloud billing data.
- Identify resources requiring investigation or manual review.
- Assess potential operational risk using CPU utilization, memory utilization, anomaly signals, and cost-related indicators.
- Classify optimization decisions into actionable categories.
- Prevent unsupported SLA compliance claims.
- Require manual approval and disable automatic resource modifications.
- Generate evaluation reports for training and testing datasets.

## 3. Module Structure

| File | Purpose |
|---|---|
| `sla_config.py` | Central configuration, thresholds, statuses, and safety settings |
| `sla_data_validation.py` | Validates required fields, identifiers, dates, and availability of SLA evidence |
| `sla_rule_engine.py` | Applies utilization, cost, and anomaly-related decision rules |
| `sla_risk_assessment.py` | Calculates heuristic risk scores, risk bands, and review priorities |
| `sla_compliance.py` | Integrates validation and risk assessment, generates compliance outputs and decision audit records |
| `sla_evaluation.py` | Checks output data quality, decision categories, and safety controls |

## 4. Input Data

The module uses the outputs of the anomaly detection stage:

- `data/anomaly_train_cloud_billing_data.csv`
- `data/anomaly_test_cloud_billing_data.csv`

Important fields include resource identifiers, service names, CPU utilization, memory utilization, usage dates, total cost, and anomaly-related indicators.

The available dataset does not contain contractual SLA targets, criticality classifications, availability measurements, or response-time measurements.

## 5. Risk Assessment

The risk assessment uses a heuristic scoring system based on selected operational and cost indicators.

| Indicator | Score contribution |
|---|---:|
| CPU utilization at or above 85% | +30 |
| CPU utilization at or above 70% but below 85% | +15 |
| Memory utilization at or above 85% | +30 |
| Memory utilization at or above 75% but below 85% | +15 |
| Anomaly indicator | +20 |
| High-cost indicator | +15 |
| Both CPU and memory utilization missing | +20 |

The total score is capped at 100.

Risk bands are assigned as follows:

- **HIGH:** Score of 50 or above
- **MEDIUM:** Score of 25 to below 50
- **LOW:** Score below 25
- **UNKNOWN:** Required utilization evidence is unavailable or invalid

These are heuristic indicators, not probabilities of SLA failure.

## 6. Optimization Decisions

The rule engine classifies resources into three categories:

- **BLOCKED:** High utilization or another configured blocking condition prevents provisional optimization.
- **REVIEW_REQUIRED:** A resource needs manual investigation before any optimization decision.
- **PROVISIONAL_CANDIDATE:** A resource may be considered for optimization assessment, subject to validation and approval.

Low utilization and high cost are optimization signals, not proof that a resource is safe to modify.

## 7. SLA Compliance Limitations

All 812 training records and 188 testing records were classified as `NOT_VERIFIABLE`.

This is intentional. Billing and utilization information alone cannot establish contractual SLA compliance or prove an SLA violation.

The current module therefore:

- Does not certify resources as SLA-compliant.
- Does not claim confirmed SLA violations.
- Does not automatically modify cloud resources.
- Requires manual approval.
- Treats anomaly signals as reasons for investigation, not proof of SLA failure.

To enable actual SLA verification, future versions must incorporate contractual targets and appropriate performance or availability measurements.

## 8. Evaluation Results

The evaluation script checked both datasets.

| Metric | Training | Testing |
|---|---:|---:|
| Resources evaluated | 812 | 188 |
| NOT_VERIFIABLE records | 812 | 188 |
| Mean heuristic risk score | 21.74 | 18.56 |
| Evaluation issues | 0 | 0 |
| Evaluation result | PASS | PASS |

The results indicate that the configured validation and safety checks passed. They do not demonstrate predictive accuracy or actual SLA compliance because ground-truth SLA labels are unavailable.

## 9. Generated Outputs

The module produces the following files in the `data/` directory:

- `sla_train_risk_assessment.csv`
- `sla_test_risk_assessment.csv`
- `sla_risk_assessment_summary.csv`
- `sla_train_compliance_results.csv`
- `sla_test_compliance_results.csv`
- `sla_compliance_summary.csv`
- `sla_decision_audit.csv`
- `sla_evaluation_summary.csv`
- `sla_evaluation_issues.csv`

These outputs support risk review, decision auditing, and evaluation.

## 10. Execution

Run the scripts from the project root in the configured Python virtual environment:

```bash
python "SLA Compliance/sla_data_validation.py"
python "SLA Compliance/sla_rule_engine.py"
python "SLA Compliance/sla_risk_assessment.py"
python "SLA Compliance/sla_compliance.py"
python "SLA Compliance/sla_evaluation.py"
```

Run the stages in order because later scripts depend on outputs from earlier stages.

## 11. Future Improvements

- Integrate contractual SLA targets and service criticality.
- Add actual availability and response-time measurements.
- Validate thresholds with domain experts and production evidence.
- Evaluate risk scores against labelled incidents when such data becomes available.
- Incorporate cloud provider pricing and resource-level performance data.
- Add approval workflows and controlled integration with cloud APIs.

## Conclusion

The SLA module provides a structured, auditable approach to assessing cloud resource risk and identifying potential optimization opportunities. It incorporates explicit safety controls and transparently reports where available data is insufficient to establish SLA compliance.
# External Dataset Model Evaluation Report

## Dataset

Microsoft FinOps Toolkit demo FOCUS billing dataset. The primary cost measure used in these analyses is `BilledCost` in USD.

## Results

### Forecasting

- Best model by holdout MAE: Naive Last Value
- Holdout MAE: 155.28930461881185 USD
- Holdout RMSE: 345.72265644917223 USD
- Holdout WAPE: 8.875446048710517%
- Test observations: 5
- The short daily history and holdout period limit confidence in generalization.
- The standalone benchmark does not automatically validate the canonical project model.

### Anomaly detection

- Records analyzed: 135272
- Cost-outlier flags: 39806
- Flagged share: 29.426636702347864%
- Flags indicate statistical billing-cost outliers, not confirmed incidents.
- No labelled incidents are available to measure precision, recall, or accuracy.

### Optimization

- High-spend review candidate groups: 63
- Candidate positive billed cost: 38813.24242645867 USD
- Candidates require usage, criticality, owner, and SLA review before any changes.
- Reduction scenarios are hypothetical and are not realized savings.

### SLA evidence

- Overall status: NOT_VERIFIABLE
- Evidence categories assessed: 10
- Missing categories: 10
- Billing data alone cannot establish uptime, latency, error rates, or contractual SLA compliance.
- Automatic production optimization must not be authorized from these billing records alone.

## Research limitations

1. The dataset is a demo sample rather than production customer evidence.
2. Forecasting evidence is limited by the short time series and holdout size.
3. Anomaly labels are unavailable, so detection accuracy cannot be established.
4. High spend does not itself prove waste, inefficiency, or safe downsizing.
5. SLA compliance remains not verifiable without validated operational measurements and targets.
6. Hypothetical cost-reduction scenarios must not be described as measured savings.

## Output folders

- `forecasting/`: holdout predictions, metrics, and forecasting graphs.
- `anomaly_detection/`: flagged records, method comparison, and cost-outlier graphs.
- `optimization/`: cost-review ranking, spend summaries, and illustrative scenarios.
- `sla/`: evidence matrix, verification statuses, and evidence graphs.
- `summary/`: consolidated metrics, inventory, overview graph, and this report.

## Overall conclusion

This external evaluation demonstrates a repeatable analysis workflow on a second billing dataset. It provides preliminary evidence about forecasting error, statistical cost outliers, spend concentration, and available SLA evidence. It does not prove production forecasting reliability, confirmed incidents, realized savings, or SLA compliance.

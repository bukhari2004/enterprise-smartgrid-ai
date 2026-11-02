# Model cards (simulated-data results)
| Model | Inputs | Output | Validation | Result | Known limits |
|---|---|---|---|---|---|
| Demand GBM (+P10/P90) | hour/dow/month, temp, cloud, region base, lag24/168, roll24 | MW per region-hour | 7 held-out days, day-ahead from midnight, recursive | 0.63 % MAPE | needs weather forecasts; lag features degrade beyond ~48 h |
| Solar/Wind GBM | clear-sky, cloud, temp / wind speed | capacity factor | last 14 days | R² 0.999/0.997 | simulated weather→power mapping is smooth |
| Transformer failure GBM | age, load, oil/winding temp, DGA H2/C2H2, moisture, vibration, faults, days since maintenance | P(fail ≤30d) | time split (months 0-17 / 18-23) | AUC 0.80, PR-AUC 0.25 (base rate 7.4 %) | labels are simulated from a logistic hazard |
| Theft blend | 11 per-meter features | suspicion 0-1 + type | 65 % of meters unlabelled hold-out | AUC 0.95 | needs real inspection labels; peers/class definitions matter |
| Battery Q-learning | hour, SoC bucket, price bucket | charge/idle/discharge | 30 held-out days | 65 % of perfect-foresight optimum | tabular, price proxy from merit order |
| Voltage / frequency | loading, feeder km, renewable ramp / loss MW, reserve | pu / Hz | analytic | n/a | simplified; use AC power flow for studies |

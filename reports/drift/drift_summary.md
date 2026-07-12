Drift simulation: reference = valid, current = distorted test (PSI alert threshold: 0.2).

| Scenario | Evidently: dataset drift | PSI alerts (features) | PSI(proba) | PR-AUC | ROC-AUC | Profit, RUB |
|---|---|---|---|---|---|---|
| no_drift | no (0/27 cols) | 0 (max 0.021: total_charges) | 0.001 | 0.671 | 0.856 | 24,500 |
| covariate | no (3/27 cols) | 3 (max 9.314: charge_diff) | 0.018 | 0.664 | 0.854 | 25,500 |
| prior | no (4/27 cols) | 0 (max 0.055: avg_service_cost) | 0.047 | 0.804 | 0.864 | 75,500 |
| concept | no (0/27 cols) | 0 (max 0.021: total_charges) | 0.001 | 0.498 | 0.614 | -8,500 |

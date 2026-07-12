Baseline policies on the validation split (n=1409):

| Policy | PR-AUC | ROC-AUC | Calls | Expected profit, RUB |
|---|---|---|---|---|
| call nobody | — | — | 0 | 0 |
| call everyone | — | — | 1409 | -848,000 |
| rule: month-to-month & tenure < 6 | — | — | 242 | -50,000 |
| LogReg on 5 features @ economic threshold | 0.640 | 0.826 | 83 | 14,500 |

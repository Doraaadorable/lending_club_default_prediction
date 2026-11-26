# Model card: Lending Club default-risk score

Following Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

| | |
|---|---|
| Model | Scikit-learn `HistGradientBoostingClassifier` inside one pipeline with its preprocessing (`loan_default.train.build_model`) |
| Hyperparameters | Best of a 40-configuration randomized search (`TUNED_PARAMS` in `train.py`; `reports/best_params.json`) |
| Output | Probability that the loan is charged off; approve if at or below the cutoff that approves the safest 80% of training loans |
| Version | Recorded in the MLflow registry (`loan-default`, alias `champion`) and in `metadata.json` with data hashes |
| Licence of the training data | CC BY 4.0 (Ariza-Garzón et al., 2024) |

## Intended use

- **Primary use:** rank personal-loan applications by default risk at application time, so a lending-policy team can choose an approval rate and see the resulting default rate.
- **Users:** credit-policy analysts, underwriters (with the explanations), model validators.
- **Out of scope:** pricing, setting loan terms, decisions about existing borrowers, and any population other than US Lending Club–style unsecured personal loans. The score must not be used without human review near the cutoff, and not without the monitoring described below.

## Factors

The audit groups we can observe are housing status, income band, census region and whether an employment length was given. Race, ethnicity, sex and age are **not** in the data, so we could not audit the protected classes directly (see Ethical considerations).

## Metrics

| Metric | Why |
|---|---|
| Default rate among approved loans at a fixed approval rate | The decision the business makes |
| ROC-AUC, average precision | Ranking quality, independent of the cutoff |
| Brier score and mean predicted against actual default | Calibration, needed when the probability itself is used |
| Approval rate, adverse impact ratio, approval among repaid borrowers, by group | Fairness |

## Evaluation data

The test set is 50,000 loans issued in 2016 (default rate 23.3%), stratified from 293,057 resolved 2016 loans. It comes from a later period than the training data, so the test measures performance on newer applications.

## Training data

The training set is 150,000 loans issued from 2007 to 2015 (default rate 18.5%), stratified from 829,347 resolved loans. Only issued loans with a final outcome appear in it.

## Quantitative results (2016 test sample)

| | Model | FICO-only baseline |
|---|---:|---:|
| ROC-AUC | 0.664 | 0.594 |
| Average precision | 0.360 | 0.282 |
| Brier score | 0.171 | 0.177 |
| Default rate, safest 80% | 19.5% | 21.6% |
| Mean predicted against actual default | 18.5% vs 23.3% | |

At the same volume of 40,000 approvals, the model approves 827 fewer loans that later defaulted (notebook 05; the current package run gives 836, from library-version differences, README section 16). The fairness audit by group is in [reports/fairness_report.md](../reports/fairness_report.md) and README section 13.

## Explanations

TreeSHAP values break each score into one contribution per application field. For any applicant, the report lists the fields that raised the score most. These are model contributions measured against an average applicant, not validated decline reasons: they are not tied to the cutoff, related fields can repeat one mechanism, and the wording has not been reviewed. The service does not return them. See `loan_default.explain` and [reports/explainability_report.md](../reports/explainability_report.md).

## Ethical considerations

- **Proxy discrimination.** ZIP code is excluded because it can stand in for race; testing showed it adds no accuracy (README section 12.2). State is still used; removing it is compared in the fairness report.
- **No protected attributes.** A real deployment must audit approval and error rates by protected class, for example using BISG proxy estimates under compliance supervision, before use.
- **Selection bias.** The model learned only from approved loans, so its risk estimates for applicant types that were rarely approved (for example FICO below 660) rest on little evidence.
- **Group-specific cutoffs.** The fairness report shows the trade-off from equal-opportunity cutoffs by income band. Cutoffs that differ by group raise legal questions under fair-lending law, especially for protected characteristics, so that mitigation is an analysis, not a policy to deploy without legal review.

## Caveats and recommendations

- Probabilities ran about 5 points low in 2016, although the ranking held (README section 11). Recalibrate on recent outcomes before using the probability itself, for example in pricing or expected-loss estimates.
- Monitor inputs weekly (PSI) and outcomes monthly by application id. Input drift alone did not detect the 2016 shift.
- Confirm the results on a later cohort that no decision in this project has touched.

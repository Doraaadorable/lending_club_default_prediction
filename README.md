# Loan Default Prediction: Balancing Approval and Risk

Approving the lowest-risk 80% of 2016 loans with gradient boosting gives a 19.5% default rate among approved loans, against 21.6% when ranking by FICO alone and 23.3% with no screening. The model ranks risk as well in 2016 as in development, but its probabilities run almost 5 points low: the applicants looked the same, yet loans of the same predicted risk defaulted more often.

```bash
python -m pip install -e ".[dev,serve,tracking]"
python -m loan_default.train          # train and save the pipeline, about 20 seconds
uvicorn loan_default.service:app      # serve predictions on port 8000
```

| Part | Where |
|---|---|
| Problem, data, analysis and results | Sections 1–10, notebooks 01–05 |
| Drift, year-by-year validation, recalibration, 2016 replay | Section 11 |
| Imputation, embeddings, clustering, semi-supervised, causal inference, AutoML, explainability | Section 12 |
| Fairness audit and mitigations | Section 13 |
| Training, tracking, registry, API, Docker, CI | Section 16 |
| Production readiness (ML Test Score) | Section 17 |
| Model card; product and data management | [docs/](docs/) |

**Data:** Ariza-Garzón et al. (2024), [Lending Club loan dataset for granting models](https://doi.org/10.5281/zenodo.11295916), CC BY 4.0.

## 1. Project Overview

Lending decisions involve two competing goals: approving enough loans to maintain lending volume while limiting defaults. A stricter policy can reduce risk, but it also rejects borrowers who would have repaid.

Our main question is:

> If we approve the lowest-risk loans, how does the default rate change as we approve more or fewer loans?

We compare five model families against a FICO-only logistic regression baseline using application-time information. We evaluate model scores alongside approval rates and default rates among approved loans.

### Main result

On the 2016 test sample, approving the lowest-risk **80%** using tuned gradient boosting produced:

- **19.50%** default rate among approved loans.
- **21.56%** with FICO alone at the same approval rate.
- **23.28%** across all test loans.

At the same volume of **40,000 approved loans**, gradient boosting retained **827 fewer loans that ultimately defaulted** than the baseline.

Risk ranking remained similar across CV and the later test cohort. However, the model underestimated 2016 default probabilities, making calibration an important remaining issue.

## 2. Business Objective and Problem Framing

### Business objective

Reduce defaults among approved loans while retaining a meaningful share of lending volume.

A policy that approves very few loans may achieve a low default rate without meeting lending-volume goals. We therefore compare models at matched approval rates and examine how risk changes as approval increases.

We ask:

- Does the model rank loan risk better than FICO alone?
- At the same approval rate, does it retain a lower-risk group?
- How many defaults are rejected, and how many fully paid loans are also rejected?

Financial returns, funding costs, and recoveries are unavailable, so we measure the approval–default trade-off rather than profit.

### Intended users and use

Credit-risk analysts could use scores to compare risk and investigate errors. Lending-policy teams could use approval curves to compare alternative policies.

The workflow is to collect application information, apply the fitted pipeline, generate risk scores, and apply a policy selected using development data.

### Machine learning task

This is a **supervised binary classification** problem.

- **Unit of analysis:** one loan.
- **Target:** `Default`.
- **Positive class, 1:** charged off.
- **Negative class, 0:** fully paid.
- **Training:** offline, using historical loans.
- **Output:** a score where higher values indicate higher default risk.

### Baseline

Manual underwriting and credit-score rules are common lending approaches. We use **FICO-only logistic regression** as a simple benchmark for assessing the value of additional application features.

## 3. Data Source and Split

**Citation.** Ariza-Garzón, M. J., Sanz-Guerrero, M., Arroyo Gallardo, J., & Lending Club. (2024). *Lending Club loan dataset for granting models* (Version 0.1) [Dataset]. Universidad Complutense de Madrid. <https://doi.org/10.5281/zenodo.11295916>. Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). `data/train.csv` and `data/test.csv` are stratified samples drawn from it (150,000 loans from 2007–2015 and 50,000 from 2016); the values are unchanged.

The [source dataset](https://doi.org/10.5281/zenodo.11295916) contains **1,347,681 loans and 15 columns**, covering origination years **2007–2018**. It retains fully paid and charged-off loans and excludes transitional outcomes.

Rejected applications are absent. Our analysis therefore concerns historical issued loans with resolved outcomes.

### Train/test split

We separate origination periods before EDA and fitted preprocessing, then take stratified samples.

| Dataset | Issue years | Available pool | Sample size | Default rate |
|---|---|---:|---:|---:|
| Training / development | 2007–2015 | 829,347 | 150,000 | 18.46% |
| Test | 2016 | 293,057 | 50,000 | 23.28% |

Sampling is stratified by `Default`, with seed **42**. Sampled train and test IDs do not overlap. EDA uses the training sample.

Sampling makes repeated comparisons manageable while approximately preserving each period’s default prevalence.

### Why split by time?

The intended task is to learn from older loans and score newer applications. The 2016 test cohort provides a later-period comparison, with observed default prevalence increasing from **18.46%** to **23.28%**.

### Why exclude 2017–2018?

Only resolved loans remain in the dataset, so recent cohorts may overrepresent loans that reached a final outcome quickly. We exclude these years to reduce this outcome-selection concern.

### Evaluation boundaries

Aggregate test prevalence was inspected in the split notebook, and the pipeline notebook documents test-period DTI anomalies. The test sample was therefore not completely unseen during development.

Model fitting, hyperparameter search, and approval-fraction selection use training data. Notebook 05 reports that its model and policy choices were fixed before final test evaluation.

Outcome-availability dates are missing. Some earlier loans may have resolved after 2016, so the split compares origination cohorts rather than fully reconstructing historically available training labels.

## 4. Exploratory Data Analysis

[Notebook 02](notebooks/02_eda.ipynb) profiles each variable’s type, missingness, unusual values, distribution, and usefulness.

### Variable summary

| Variable | Meaning | Treatment |
|---|---|---|
| `id` | Loan identifier | Overlap checks; excluded from predictors |
| `issue_d` | Issue month and year | Temporal splitting and cohort analysis |
| `revenue` | Self-reported annual income | Log transformation and standardization |
| `dti_n` | Monthly debt-to-income ratio, excluding mortgage, as defined by the source | Cleaning, median imputation and standardization |
| `loan_amnt` | Requested loan amount | Log transformation and standardization |
| `fico_n` | Midpoint of reported FICO range | Median imputation and standardization |
| `experience_c` | Borrower-experience indicator | Excluded because it is nearly constant |
| `emp_length` | Employment length | Numeric years, imputation and missingness indicator |
| `purpose` | Loan purpose | One-hot encoding with rare-category grouping |
| `home_ownership_n` | Housing status | One-hot encoding with rare-category grouping |
| `addr_state` | State of residence | Smoothed target encoding with cross-fitting |
| `zip_code` | Partially masked ZIP code | Excluded |
| `title` | Borrower-provided loan title | Excluded because of high cardinality and overlap with purpose |
| `desc` | Borrower-provided loan description | Excluded because of missingness and changing availability |
| `Default` | Charged off or fully paid | Target |

### Main findings

- Approximately **18.5%** of training loans defaulted.
- Income has skew around **46.4** and a maximum of **USD 9.5 million**.
- Log transformation reduces income skew to approximately **0.18**.
- About **5.2%** have `emp_length = NI`, with an observed default rate of approximately **23.7%**.
- Only **5 of 150,000** training observations have `experience_c = 0`.
- ZIP code has **879 categories**; approximately **20%** have fewer than 30 observations.
- Loan descriptions are approximately **86% missing** and almost absent in 2015.
- Only **90 training observations** have FICO below 660.

### Charts and takeaways

| Chart | Main takeaway |
|---|---|
| Default rate by FICO | Higher FICO bands generally have lower default rates |
| Default rate by purpose | Small-business loans have higher observed default rates than car loans |
| Default rate by home ownership | Renters have higher observed default rates than mortgage holders |
| Default rate by year | Rates vary over time and rise in later training years |
| Default rate by loan-to-income | Higher ratio groups have higher observed default rates |

Individual numeric rank correlations with default are weak. We therefore test combinations of features and evaluate their contribution through validation.

### Useful additional data

| Additional data | Purpose |
|---|---|
| Income verification | Assess reporting reliability |
| Dates when outcomes became known | Reconstruct historically available labels |
| Recent cohorts with sufficient follow-up | Assess current relevance |
| Loan returns, costs and recoveries | Evaluate financial value |
| Broader application records | Understand coverage beyond approved applicants |
| Appropriate protected-group information | Assess approval and error disparities directly |

## 5. Preprocessing and Feature Engineering

[Notebook 03](notebooks/03_pipeline.ipynb) explains the shared pipeline in [preprocessing.py](src/loan_default/preprocessing.py).

Full models use eight original application inputs and the engineered `loan_to_income` feature.

### Cleaning

- Convert employment length to numeric years from 0 to 10.
- Map `NI` to missing and preserve missingness through an indicator.
- Treat DTI equal to 999 as missing.
- Cap DTI at 100.
- Calculate `loan_to_income = loan_amnt / revenue`.
- Drop identifiers, issue date, experience indicator, ZIP code, title and description.

The source does not explain DTI equal to 999. Treating it as missing is an explicit assumption. The training sample has no such values and no DTI above 100; these rules handle later-period observations.

### Transformations

| Group | Features | Transformation |
|---|---|---|
| Numeric | FICO, DTI, employment length | Median imputation with missing indicators, then standardization |
| Skewed numeric | Income, loan amount, loan-to-income | `log(1 + x)`, then standardization |
| Categorical | Purpose, housing status | One-hot encoding; categories with fewer than 500 observations grouped as infrequent |
| Geographic | State | Target encoding with smoothing of 100 and internal cross-fitting |

Income percentile clipping and FICO bands were not included in the final shared pipeline.

The full-training pipeline check produces **24 numeric columns**, with no remaining missing values.

### Leakage control

Learned preprocessing is fitted inside each CV training fold. State target encoding uses internal cross-fitting so a row’s own target does not determine its training encoding.

The FICO-only baseline uses a separate pipeline with FICO imputation and scaling.

## 6. Evaluation Metrics

### ROC-AUC

ROC-AUC measures how well a model ranks defaulted loans above fully paid loans. It is the primary metric for model comparison and hyperparameter selection.

### Average Precision

We report **Average Precision (AP)** using `average_precision_score`, alongside precision–recall analysis.

AP focuses on default identification in an imbalanced dataset. The prevalence reference is **18.46%** in training and **23.28%** in testing. AP differs from trapezoidal PR-AUC, so we label it explicitly.

### Approval and default rates

`Approval rate = approved loans / evaluated loans`

`Approved-loan default rate = defaults among approved loans / approved loans`

Models are compared at the same approval rate, so lower risk is not achieved simply by approving fewer loans. Approval here means retaining loans from the historical issued-loan sample.

### Supporting diagnostics

We examine confusion matrices, default precision and recall, Brier score, and calibration.

Accuracy alone is insufficient: predicting every training loan as fully paid achieves approximately **81.5% accuracy** while identifying no defaults.

## 7. Model Comparison and Shortlist

[Notebook 04](notebooks/04_modeling_baseline.ipynb) compares five model families against the FICO-only baseline.

All models use **stratified five-fold CV**, with shuffling and seed **42**. Learned preprocessing is fitted separately within each fold. Results are mean ± sample standard deviation.

| Model | CV ROC-AUC | CV AP |
|---|---|---|
| Gradient boosting | 0.6627 ± 0.0047 | 0.3007 ± 0.0084 |
| Logistic regression | 0.6610 ± 0.0047 | 0.2965 ± 0.0083 |
| Random forest | 0.6608 ± 0.0041 | 0.2970 ± 0.0065 |
| Extra Trees | 0.6554 ± 0.0039 | 0.2930 ± 0.0087 |
| Decision tree | 0.6420 ± 0.0044 | 0.2736 ± 0.0057 |
| FICO-only baseline | 0.5904 ± 0.0043 | 0.2254 ± 0.0028 |

All multivariable models outperform the FICO-only baseline.

We shortlisted:

1. **Gradient boosting:** highest mean ROC-AUC and AP.
2. **Logistic regression:** similar performance with a simpler model.
3. **Random forest:** a competitive alternative ensemble approach.

The small differences between these candidates motivate examining tuning gains and complexity.

Shuffled CV mixes origination years. The separate 2016 test provides the later-cohort evaluation, and section 11 adds a year-by-year chronological check.

## 8. Feature Importance and Error Analysis

### Important variables

Held-out permutation importance uses up to 3,000 observations from the first validation fold and five permutations per feature.

For gradient boosting:

| Original input | Mean ROC-AUC reduction after permutation |
|---|---:|
| FICO | 0.0673 |
| Loan amount | 0.0657 |
| Income | 0.0340 |
| DTI | 0.0264 |

Shuffling loan amount or income also changes the derived loan-to-income feature. These values measure predictive contribution through the pipeline, rather than causal effects.

### Default threshold

At the preliminary threshold of **0.50**, untuned gradient boosting identified only **82 of 27,684** training defaults in out-of-fold predictions.

Most scores fall below 0.50. This produces high accuracy but very low default recall, motivating a policy based on approval volume.

### Business errors

With default as the positive class:

- **False approval:** approve a loan that defaults — false negative.
- **False rejection:** reject a loan that repays — false positive.

Notebook 04 examines errors across FICO, predicted risk, housing, purpose, employment information, and other loan characteristics. Verified income and more detailed repayment obligations would help investigate individual errors.

## 9. Hyperparameter Tuning and Ensembles

[Notebook 05](notebooks/05_tuning_and_final_evaluation.ipynb) tunes the three shortlisted models using `RandomizedSearchCV` and the same five-fold design.

### Search budgets

| Model | Configurations | Main parameters |
|---|---:|---|
| Gradient boosting | 40 | Learning rate, iterations, leaf count, minimum leaf size, regularization, feature fraction |
| Random forest | 20 | Tree count, depth, minimum leaf size, feature fraction |
| Logistic regression | 15 | Regularization strength and L1/L2 penalty |

ROC-AUC determines the selected configuration; AP is also recorded. Randomized search explores broad ranges within a manageable number of fits.

### Tuning results

| Model | Untuned CV ROC-AUC | Tuned CV ROC-AUC | Tuned AP |
|---|---:|---|---:|
| Gradient boosting | 0.6627 | 0.6636 ± 0.0046 | 0.3012 |
| Random forest | 0.6608 | 0.6612 ± 0.0039 | 0.2973 |
| Logistic regression | 0.6610 | 0.6610 ± 0.0047 | 0.2966 |

The largest tuning gain is approximately **0.0009 ROC-AUC**. Search adds little compared with the improvement from using more information than FICO alone.

### Final gradient boosting configuration

The final estimator is `HistGradientBoostingClassifier`, with settings rounded here for readability:

| Parameter | Value |
|---|---:|
| `learning_rate` | 0.0249 |
| `max_iter` | 510 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 337 |
| `l2_regularization` | 0.0257 |
| `max_features` | 0.5823 |

Full-precision values are in [reports/best_params.json](reports/best_params.json), and all 40 configurations in [reports/search_results.csv](reports/search_results.csv); `python -m loan_default.search` reruns the search. Early stopping can stop fitting before the maximum iteration count.

### Ensembles

We tested soft voting and stacking with a logistic-regression meta-model and internal five-fold CV.

Both achieved approximately **0.6642 CV ROC-AUC**, versus **0.6636** for tuned gradient boosting, without an AP improvement. Component predictions were highly correlated, approximately **0.93–0.97**.

We retained gradient boosting because the small gain did not justify running three models. The notebook required a gain larger than one fold standard deviation as a practical complexity rule, not a significance test.

## 10. Approval Policy and Final Test Results

### Policy selection

Using out-of-fold training predictions, we selected **80% approval** before final test evaluation.

We evaluated:

1. **Fixed approval fraction:** approve the lowest-risk 80% of the scored batch.
2. **Fixed training cutoff:** apply the threshold derived from training predictions.

Score ties in the fixed-fraction policy are broken randomly with seed **42**, particularly important for the FICO baseline.

At 80% training approval, approved-loan default rates were **15.06%** for gradient boosting and **17.08%** for FICO. Training-derived cutoffs were **0.2565** and **0.2281**, respectively.

The final model and baseline were fitted on all 150,000 training loans. Notebook 05 reports that predictions were saved after final test scoring and no further tuning followed.

### Predictive performance

| Model | Test ROC-AUC | Test AP | Test Brier |
|---|---:|---:|---:|
| FICO-only baseline | 0.5943 | 0.2822 | 0.1774 |
| Tuned gradient boosting | 0.6639 | 0.3598 | 0.1706 |

Gradient boosting improves test ROC-AUC by approximately **0.0696**. Its test score is close to its **0.6636** tuned CV result, showing similar ranking performance on the later cohort.

### Approval and default rates

| Policy | Approval rate | Approved loans | Approved-loan default rate |
|---|---:|---:|---:|
| Retain all test loans | 100.00% | 50,000 | 23.28% |
| FICO-only, safest 80% | 80.00% | 40,000 | 21.56% |
| Gradient boosting, safest 80% | 80.00% | 40,000 | 19.50% |
| FICO-only, training cutoff | 81.24% | 40,621 | 21.70% |
| Gradient boosting, training cutoff | 80.07% | 40,036 | 19.50% |

At matched 80% approval, gradient boosting lowers the default rate by approximately **2.07 percentage points** relative to FICO alone.

At 50% approval, test default rates are **14.86%** for gradient boosting versus **18.40%** for FICO. At 90%, they are **21.08%** versus **22.48%**. Gradient boosting has lower observed default rates at each evaluated approval fraction from 10% to 90%.

### Errors at 80% approval

| Outcome | Gradient boosting | FICO-only |
|---|---:|---:|
| Approved and fully paid | 32,201 | 31,374 |
| Approved and defaulted | 7,799 | 8,626 |
| Rejected and fully paid | 6,158 | 6,985 |
| Rejected and defaulted | 3,842 | 3,015 |

Gradient boosting rejects **33.0%** of defaults, compared with **25.9%** for FICO.

However, **61.58%** of rejected loans were fully paid. Better ranking reduces errors relative to the baseline but still excludes many borrowers who repay.

### Bootstrap uncertainty

Notebook 05 uses **1,000 paired bootstrap resamples** of the test sample. The 95% percentile interval for the approved-loan default-rate difference, gradient boosting minus FICO at 80% approval, is approximately **−2.37 to −1.85 percentage points**.

This describes uncertainty within the observed test sample, not retraining variability or future changes in lending conditions.

## 11. Calibration and Temporal Change

Ranking remained stable, but calibration changed.

On training out-of-fold predictions, predicted and observed default rates were within approximately **0.7 percentage points** across ten risk groups.

On the 2016 test sample:

- Mean predicted default probability: **18.49%**
- Actual default prevalence: **23.28%**

The model underestimated defaults across all ten test risk groups.

Risk ranking and probability accuracy therefore need separate evaluation. The model identifies relatively safer loans, while its numerical probabilities understate the later cohort’s default level.

### What changed between the periods

[`python -m loan_default.drift`](src/loan_default/drift.py) compares the 2016 sample with the training sample and writes [reports/drift_report.md](reports/drift_report.md).

**The inputs barely changed.** The population stability index (PSI) is below 0.04 for all nine model features, against a usual threshold of 0.10 for a moderate shift. The largest values are for state (0.038) and loan purpose (0.036). Every two-sample test rejects equality (p < 0.01), but with 150,000 and 50,000 loans the tests detect differences too small to matter, so drift is read from PSI.

**Loans of the same predicted risk defaulted more often.** Grouping loans into deciles of predicted risk, the 2016 default rate is higher in every band, by 1.7 percentage points in the safest band and 5.3 to 7.4 points in the riskier half.

| | Default rate |
|---|---:|
| Training sample | 18.46% |
| 2016, if only the mix of risk bands had changed | 18.47% |
| 2016, actual | 23.28% |

Of the 4.83-point increase, 0.01 points come from a different mix of applicants and 4.82 points from higher default rates among applicants the model scores alike. In drift terms this is a change in the relationship between inputs and outcome (concept drift), not a change in the inputs (covariate drift). It explains the pattern above: ranking held because the ordering of risk was preserved, while probabilities were too low because the overall level moved.

The increase continues a trend inside the training period: 15.49% for 2013 originations, 18.13% for 2014, 20.32% for 2015. The data cannot separate two explanations. Credit quality may have worsened. Alternatively, because only resolved loans are kept and defaults resolve sooner than full repayment, recent cohorts may over-represent defaults (section 14). Monitoring input distributions alone would not have flagged this change; detecting it needs outcome labels.

### Would chronological validation have shown it?

[`python -m loan_default.temporal`](src/loan_default/temporal.py) scores each year with a model trained only on earlier years ([reports/temporal_validation.md](reports/temporal_validation.md)).

| Evaluated year | Trained on | ROC-AUC | Mean predicted | Actual | Gap |
|---|---:|---:|---:|---:|---:|
| 2012 | 7,696 | 0.6275 | 17.95% | 16.65% | −1.30 pp |
| 2013 | 17,290 | 0.6320 | 18.07% | 15.49% | −2.59 pp |
| 2014 | 41,842 | 0.6474 | 16.89% | 18.13% | +1.23 pp |
| 2015 | 82,077 | 0.6669 | 17.81% | 20.32% | +2.52 pp |
| 2016 | 150,000 | 0.6639 | 18.49% | 23.28% | +4.79 pp |

Shuffled 5-fold CV on 2007–2015 gives ROC-AUC 0.6633 and a mean prediction exactly equal to the observed rate (18.46%).

**Ranking:** shuffled CV did not overstate it. Chronological ROC-AUC rises with the size of the training set and reaches 0.667 for 2015, in line with the shuffled estimate and the 2016 result.

**Calibration:** shuffled CV hid the problem. Because every fold mixes all years, predicted and actual rates match by construction. Scored year by year, the model overpredicts in 2012–2013, when defaults fell, and underpredicts from 2014, by 1.2 points and then 2.5 points. The 4.8-point gap in 2016 continues a trend that this check shows in the development data.

**Tuning on year-ordered folds does not fix it.** `python -m loan_default.search --folds chronological` reruns the 40-configuration search with one fold per year from 2012 to 2015, each trained on earlier years only ([reports/search_results_chronological.csv](reports/search_results_chronological.csv)). It picks a different configuration: a lower learning rate (0.0125) and smaller trees (7 leaves). On 2016:

| Tuned on | Test ROC-AUC | Test AP | Brier | Mean predicted | Default rate, safest 80% |
|---|---:|---:|---:|---:|---:|
| Shuffled folds (current model) | 0.6639 | 0.3599 | 0.1706 | 18.49% | 19.48% |
| Year-ordered folds | 0.6621 | 0.3585 | 0.1712 | 18.39% | 19.53% |

The two are within the spread of the search itself (the top five chronological configurations lie within 0.002 ROC-AUC), and neither moves the mean prediction toward the 23.28% observed. Hyperparameters set how risk is ranked, not the level the model expects, so the current model is kept and the calibration gap is handled by recalibration below.

### Does recalibration help?

[`python -m loan_default.recalibration`](src/loan_default/recalibration.py) fits a correction on loans issued in January–June 2016 and judges it on loans issued in July–December 2016 ([reports/recalibration_report.md](reports/recalibration_report.md)).

| Method | Mean predicted | Actual | Brier | Mean gap over 10 risk bins | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| No recalibration | 18.11% | 24.59% | 0.1794 | 6.49 pp | 0.6587 |
| Intercept shift | 21.67% | 24.59% | 0.1757 | 2.93 pp | 0.6587 |
| Platt scaling | 21.64% | 24.59% | 0.1758 | 2.95 pp | 0.6587 |
| Isotonic | 21.66% | 24.59% | 0.1758 | 2.93 pp | 0.6578 |

Recalibration removes a little over half of the gap. The three methods give the same result, so the error is a shift in level: one constant on the log-odds scale does as well as a flexible mapping. The remaining gap exists because the default rate kept rising within 2016, from 22.39% in the first half to 24.59% in the second, so a correction fitted on one period is already behind in the next.

Ranking is unchanged, so the approval policy and its results in section 10 are unaffected; recalibration matters where the probability itself is used, for example in pricing or loss forecasts.

In production the first half's outcomes would not be known by the second half, so a live correction would use older labels and lag further.

### Replaying 2016 as if the model were live

[`python -m loan_default.replay`](src/loan_default/replay.py) trains on 2007–2015, scores each 2016 loan in the month it was issued, writes the predictions in the service's log format, and runs both monitors on them ([reports/replay_report.md](reports/replay_report.md)).

| Month | Largest input PSI | Mean predicted | Actual | Gap |
|---|---:|---:|---:|---:|
| 2016-01 | 0.053 | 18.34% | 19.96% | +1.62 pp |
| 2016-02 | 0.037 | 18.94% | 19.52% | +0.58 pp |
| 2016-03 | 0.066 | 19.15% | 22.19% | +3.04 pp |
| 2016-04 | 0.051 | 18.43% | 25.10% | +6.67 pp |
| 2016-05 | 0.058 | 18.98% | 25.70% | +6.72 pp |
| 2016-06 | 0.064 | 18.27% | 25.05% | +6.78 pp |
| 2016-07 | 0.085 | 18.60% | 26.14% | +7.54 pp |
| 2016-08 | 0.086 | 18.38% | 25.35% | +6.97 pp |
| 2016-09 | 0.092 | 18.02% | 25.15% | +7.13 pp |
| 2016-10 | 0.088 | 17.51% | 23.38% | +5.87 pp |
| 2016-11 | 0.091 | 17.80% | 22.66% | +4.86 pp |
| 2016-12 | 0.092 | 18.19% | 24.53% | +6.34 pp |

**The input monitor stays quiet while the model is wrong.** No month reaches PSI 0.10. The largest monthly value climbs from about 0.05 to 0.09, driven from July by the state mix; random training samples of one month's size score about 0.02 on state, so the shift is real but small. Run on the replayed log, the weekly drift check passes.

**The outcome monitor catches it.** [`loan_default.monitor`](src/loan_default/monitor.py) joins outcomes to predictions by application id and flags any month whose actual default rate is more than 2 points from the predicted rate. It flags every month from March, and its command exits with an error.

**How early it fires depends on how soon outcomes are known.** The delays below are assumptions about when a lender knows a month's default rate well enough to compare; the dataset records only the final status. Recalibration shifts the log-odds by a constant fitted on the 2016 months whose outcomes are already known.

| Outcomes known after | Outcome alert fires | Mean monthly gap (points), no recalibration | Quarterly | Monthly |
|---|---|---:|---:|---:|
| 3 months | June 2016 | 5.3 | 3.3 | 3.1 |
| 6 months | September 2016 | 5.3 | 4.3 | 4.2 |
| 12 months | March 2017 | 5.3 | 5.3 | 5.3 |

With a three-month delay, monthly recalibration cuts the average gap from 5.3 to 3.1 points; with a twelve-month delay nothing can be corrected within the year. The speed of outcome data, not the model, sets how quickly this kind of error is caught. An earlier signal such as missed first payments would shorten the delay, but this dataset does not record payments.

### The four kinds of drift, and where each is measured

| Drift | Question | Where | 2016 finding |
|---|---|---|---|
| Feature drift | Did the inputs change? | `drift.py` section 1 (PSI, KS and chi-squared per feature); weekly in `drift.yml` | Stable: every PSI below 0.04 |
| Target drift | Did the default rate change? | `drift.py` section 2 (default rate by issue year); `monitor.py` by month | Yes: 18.5% to 23.3%, continuing a rise from 2013 |
| Concept drift | Do loans of the same predicted risk default more often? | `drift.py` sections 3–4 (default rate within risk bands; gap split into mix and within-band change) | Yes: of the +4.83-point rise, +4.82 comes from higher default rates within bands and +0.01 from a different risk mix |
| Prediction drift | Did the distribution of scores change? | `replay.py` (mean predicted per month); service log | Small: monthly mean prediction stays between 17.5% and 19.2% |
| Model performance | Are ranking and calibration holding? | `temporal.py` year by year; `monitor.py` calibration gap per month, with alert | Ranking holds (ROC-AUC 0.664); calibration gap of +4.8 points |
| Hyperparameters | Which settings produced the served model? | MLflow run parameters, `metadata.json`, registry version | Recorded for every training run |

## 12. Further Techniques: Imputation, Embeddings, Clustering, Semi-Supervised Learning, Causal Inference, AutoML and Explainability

Each technique below is a module under `src/loan_default/` with unit tests. Each one writes a report to `reports/`, and the numbers here come from those reports. Every model comparison uses the same harness ([experiments.py](src/loan_default/experiments.py)): shuffled 5-fold CV on 2007–2015, then a refit scored on 2016. Gains in test ROC-AUC below about 0.002 fall within the noise of one cohort.

| Technique | Module | Report | Headline |
|---|---|---|---|
| Advanced imputation | `imputation` | [imputation_report.md](reports/imputation_report.md) | MICE recovers hidden values best; the missing flag matters more than the fill value |
| Entity embeddings | `embedding` | [embedding_report.md](reports/embedding_report.md) | Boosting on learned embeddings: +0.0008 test ROC-AUC; adding ZIP makes it worse |
| PCA and clustering | `unsupervised` | [unsupervised_report.md](reports/unsupervised_report.md) | Four readable borrower segments, default rates 9%–26%; no accuracy gain as features |
| Semi-supervised learning | `semi_supervised` | [semi_supervised_report.md](reports/semi_supervised_report.md) | Small gains (≤ 0.004 ROC-AUC) only when 1–5% of outcomes are known; labels matter far more than method |
| Causal inference | `causal` | [causal_report.md](reports/causal_report.md) | A loan of at least 25% of income raises default by about 9 points (AIPW) |
| Bayesian HPO and AutoML | `hpo` | [hpo_report.md](reports/hpo_report.md) | TPE finds good settings faster than random search; the response surface is flat, so production keeps its parameters |
| Explainability | `explain` | [explainability_report.md](reports/explainability_report.md) | FICO and loan size relative to income drive scores; reason codes for each declined applicant |
| Fairness audit | `fairness` | [fairness_report.md](reports/fairness_report.md) | Section 13 |

```bash
python -m pip install -e ".[research]"     # optuna, shap, matplotlib
python -m loan_default.imputation           # about 7 minutes on four cores
python -m loan_default.embedding            # about 1 minute
python -m loan_default.unsupervised         # about 1 minute
python -m loan_default.semi_supervised      # about 7 minutes
python -m loan_default.causal               # about 2 minutes
python -m loan_default.hpo [--track]        # about 10 minutes; --track logs every trial to MLflow
python -m loan_default.explain              # about 1 minute
python -m loan_default.fairness             # about 2 minutes
```

### 12.1 Advanced imputation

After cleaning, the only gaps in training data are in employment length (5.2% missing). These gaps are not random: **23.7%** of loans without an employment length defaulted, against **18.2%** of loans with one. We compared mean and median filling with two model-based imputers. **Iterative (MICE)** predicts each column from all the others with chained regressions. **KNN** averages the 10 most similar loans. The model-based imputers see income and loan size, so these can inform a missing value.

*Recovering values hidden at random* (10% of known FICO, DTI and employment values, 30,000-loan sample), RMSE:

| Column | Median | Mean | Iterative (MICE) | KNN |
|---|---:|---:|---:|---:|
| FICO | 30.08 | 29.91 | **29.59** | 30.88 |
| DTI | 8.18 | 8.17 | **7.79** | 8.16 |
| Employment years | 3.64 | 3.64 | **3.61** | 3.81 |

*Effect on the model:*

| Imputer | CV ROC-AUC | Test ROC-AUC | Test default rate, safest 80% |
|---|---:|---:|---:|
| Median + missing flag (production) | 0.6633 | 0.6639 | 19.48% |
| Iterative (MICE) + missing flag | 0.6634 | 0.6641 | 19.41% |
| KNN + missing flag | 0.6635 | 0.6637 | 19.49% |
| Median, no flag | 0.6630 | 0.6628 | 19.46% |
| Iterative (MICE), no flag | 0.6628 | 0.6625 | 19.51% |

MICE is the best at recovering values, but the application fields are only weakly related to each other, so even MICE barely beats the median. For the model, the **missing-value flag** is what counts. Removing it costs about 0.001 test ROC-AUC whichever imputer is used, because the fact that a borrower gave no employment length is itself a risk signal (missing not at random). MICE with the flag gains 0.0002, which is within the noise, so production keeps the simpler median imputation. `build_preprocessor(numeric_imputer="iterative")` switches imputers if a future data source has more gaps.

### 12.2 Entity embeddings

[embedding.py](src/loan_default/embedding.py) learns a short vector for each state, loan purpose, housing status and ZIP prefix while predicting default (Guo & Berkhahn, 2016). The network is written in NumPy, so the project needs no deep-learning framework. Each categorical field gets an embedding of size min(10, levels/2), and levels with fewer than 30 loans share a "rare or unseen" vector. The embeddings and the scaled numeric fields feed one ReLU layer of 32 units and then a sigmoid output. Training uses Adam with weight decay and early stopping on a held-out 10%. A unit test checks every gradient against finite differences.

| Model | CV ROC-AUC | Test ROC-AUC | Test AP | Test default rate, safest 80% |
|---|---:|---:|---:|---:|
| One-hot + target encoding (production) | 0.6633 | 0.6639 | 0.3599 | 19.48% |
| Embedding network on its own | 0.6627 | 0.6637 | 0.3577 | 19.38% |
| Boosting on learned embeddings | **0.6638** | **0.6647** | **0.3604** | 19.45% |
| Embedding network, with ZIP | 0.6619 | 0.6626 | 0.3542 | 19.41% |
| Boosting on learned embeddings, with ZIP | 0.6584 | 0.6607 | 0.3528 | 19.51% |

- **A one-layer network matches gradient boosting** (0.6637 against 0.6639): on this data the signal is mostly smooth and additive.
- **Embeddings as boosting inputs give the best single model** (+0.0008 test ROC-AUC, +0.0005 CV). The gain is consistent but small.
- **ZIP code does not help.** Its 879 prefixes, many with few loans, let both models fit noise, and test ROC-AUC falls by up to 0.003. Excluding ZIP for fairness reasons (section 13) therefore costs nothing in accuracy.
- **The embeddings are interpretable.** The first principal component of the state embeddings correlates **0.90** with each state's default rate, so the network recovers what target encoding captures. The remaining dimensions group states with similar borrowers ([state_embeddings.png](reports/figures/state_embeddings.png)).

### 12.3 Dimensionality reduction and clustering

**PCA.** The 23 preprocessed features show little redundancy. The first component explains only 10.1% of the variance (housing status and income), the second 8.5% (debt consolidation and loan-to-income), and 95% of the variance needs 19 components. A model trained on those 19 components loses 0.0016 test ROC-AUC (0.6624), because rotating the features breaks the axis-aligned splits that trees use. PCA is useful here for description, not as model input.

**K-means clustering** on the six-field financial profile (FICO, DTI, employment length, log income, log loan, log loan-to-income). Silhouette scores are low (0.18 at k = 2, falling slowly), so borrowers form a continuum rather than distinct groups. We use k = 4, the largest k within 0.01 of the best silhouette, as a readable segmentation:

| Segment | Share | Mean FICO | Median income | Loan / income | Median DTI | Default 2007–15 | Default 2016 |
|---|---:|---:|---:|---:|---:|---:|---:|
| High FICO, low DTI | 12.0% | 757 | $74,000 | 0.20 | 13.7 | 8.7% | 11.1% |
| Small loans, short employment | 27.7% | 686 | $50,000 | 0.12 | 16.9 | 15.7% | 20.8% |
| Higher income, long employment | 27.0% | 690 | $100,000 | 0.17 | 13.6 | 15.9% | 21.4% |
| Large loan for income, high DTI | 33.3% | 691 | $51,000 | 0.33 | 23.2 | 26.3% | 32.5% |

The segments separate default risk threefold, and their order is unchanged in 2016. Every segment's default rate rose by 2–6 points, which matches the uniform level shift found in section 11. The riskiest segment is defined by debt burden, not credit score: its mean FICO is the same as the middle two segments'. This makes the segments useful for policy discussions and monitoring. Adding the distances to the four centres as model features changes test ROC-AUC by +0.0001, because the boosting model already finds these interactions.

### 12.4 Semi-supervised learning

In practice many applications never get an outcome. Rejected applicants have none (the *reject inference* problem), and recent loans are still running. [semi_supervised.py](src/loan_default/semi_supervised.py) simulates this: it hides the outcome of all but 1%, 5% or 20% of the 2007–2015 loans, trains six methods with the same early-stopped gradient boosting learner, and scores each on the fully labeled 2016 sample. Each setting is repeated with three random labeled subsets.

| Method | 1% labeled (~1,500) | 5% labeled (~7,500) | 20% labeled (~30,000) |
|---|---:|---:|---:|
| Labeled only | 0.6086 (±0.0085) | 0.6425 (±0.0043) | **0.6568** (±0.0013) |
| Self-training (scikit-learn, threshold 0.9) | 0.6062 (±0.0021) | 0.6430 (±0.0016) | 0.6555 (±0.0005) |
| Balanced self-training | **0.6123** (±0.0061) | 0.6404 (±0.0027) | 0.6552 (±0.0010) |
| Cluster soft labels (k-means) | 0.6117 (±0.0025) | **0.6445** (±0.0015) | 0.6553 (±0.0006) |
| Label spreading (10-NN graph) | 0.6046 (±0.0031) | 0.6355 (±0.0024) | 0.6533 (±0.0009) |
| Cluster features | 0.6085 (±0.0059) | 0.6402 (±0.0037) | 0.6552 (±0.0011) |

Test ROC-AUC, mean (± standard deviation) over three labeled subsets. With every outcome known, the production model reaches 0.6639.

- **Labels matter far more than method.** Going from 1% to 20% labeled adds 0.05 ROC-AUC. The best semi-supervised method adds at most 0.004 at 1% and 0.002 at 5%, which is within one standard deviation of the labeled-only result.
- **Plain self-training drifts.** Default probabilities rarely exceed 0.9, so scikit-learn's thresholded self-training adds almost only "repaid" pseudo-labels, and this skews the training mix. The balanced variant, which adds confident defaults and repayments in the labeled base rate, is the best method at 1%.
- **Cluster soft labels are the most stable gain.** Giving each unlabeled loan its cluster's smoothed default rate helps at 1% and 5% and has the lowest variance across subsets.
- **With 20% labeled (30,000 loans), every method falls slightly behind using the labels alone.** Pseudo-labels then add more noise than information.
- **The lesson for reject inference:** unlabeled applications help only when outcomes are very scarce, and only a little, because default depends weakly on the features the unlabeled loans share. Outcomes for even a small random sample of otherwise-rejected applicants (a "test-and-learn" approval budget) would be worth more than any of these methods.

### 12.5 Causal inference: does a large loan cause default?

The model shows that loans which are large relative to income default more often. A policy question needs more than that: would *capping* the loan amount reduce defaults? [causal.py](src/loan_default/causal.py) estimates the average treatment effect of a loan of at least **25% of annual income** (36.9% of loans). It adjusts for everything known at application: FICO, DTI, employment length, income, purpose, housing, state and issue year. All nuisance models use 5-fold cross-fitting.

| Estimator | 2007–2015 | 2016 (replication) |
|---|---:|---:|
| Naive difference | +9.90 pp | +10.47 pp |
| Regression adjustment (T-learner) | +8.98 pp | +9.30 pp |
| Inverse propensity weighting | +9.55 pp | +9.76 pp |
| **Doubly robust (AIPW)** | **+9.12 pp** (95% CI 8.68–9.56) | **+9.56 pp** (95% CI 8.66–10.46) |

- **Confounding is modest.** Adjusting removes about 0.8 points of the raw 9.9-point gap. Borrowers who take large loans have higher DTI (standardized difference 0.45) and lower income (−0.63), which explains part of their risk. After weighting, every covariate is within 0.05 except income (−0.17): with income in the treatment's definition, overlap on income is limited (6.7% of propensities fall outside 0.05–0.95). AIPW stays consistent through this because its outcome model adjusts for income directly.
- **Refutation tests pass.** A shuffled placebo treatment gives −0.30 pp (CI includes 0). Adding a random confounder leaves the estimate at +9.14 pp. A random half of the data gives +9.19 pp.
- **The effect is larger for weaker credit:** +10.9 pp below FICO 680, +8.9 pp for 680–719, +6.5 pp at 720 and above. A cap on loan-to-income would do most good in the lowest band.
- **Caveat.** Lending Club set the interest rate, grade and term (36 or 60 months) after seeing the application, and the data records none of them. Larger loans more often carry 60-month terms and higher rates. The estimate is therefore the effect of a large loan *together with the terms that come with it*, under the assumption of no unrecorded confounding. A randomized cap, or data with term and rate, would be needed to separate them.

### 12.6 Bayesian hyperparameter optimization and AutoML

[hpo.py](src/loan_default/hpo.py) adds two techniques beyond the randomized search of section 9. With `--track`, every trial is logged to MLflow as a nested run under its study.

**1. Bayesian optimization (Optuna TPE) with pruning.** TPE uses the same six hyperparameters, ranges, 40-configuration budget and 5-fold CV as the randomized search. Each trial reports its running CV score after every fold, and a median pruner stops trials that fall clearly behind; 6 of the 40 trials were stopped early. The study took 7.4 minutes on four cores.

| | Randomized search | TPE |
|---|---:|---:|
| Best CV ROC-AUC | 0.6634 (as rerun by `loan_default.search`) | **0.6638** |
| Median CV ROC-AUC of the trials | 0.6625 | **0.6635** |
| Best after 5 trials | — | 0.6635 |
| Test ROC-AUC of the winner (2016) | **0.6639** | 0.6636 |

TPE concentrates its trials in the good region: its median trial scores 0.0010 higher than the random search's median, and after 5 trials it already beats the random search's best. The final winners differ by 0.0004 in CV and 0.0003 on test, in opposite directions. The response surface is flat, which confirms section 9's conclusion that tuning cannot add much here. fANOVA importance over the completed trials ranks L2 regularization (0.29), tree size (0.21) and the number of iterations (0.18) above the learning rate (0.03).

**2. AutoML as a CASH problem** (combined algorithm selection and hyperparameter optimization). One TPE study chooses the model family *and* its settings: gradient boosting, random forest, extra trees, logistic regression or the entity-embedding network. To fit the budget it ran 40 trials on a 50,000-loan sample with 3-fold CV (1.7 minutes).

| Family | Trials | Best CV ROC-AUC (50k sample) |
|---|---:|---:|
| Embedding network | 15 | **0.6575** |
| Logistic regression | 10 | 0.6558 |
| Gradient boosting | 7 | 0.6551 |
| Random forest | 4 | 0.6525 |
| Extra trees | 4 | 0.6523 |

TPE spent most of its budget on the two best families. On 50,000 loans the smooth models (the network and logistic regression) beat the trees. Refitted on all 150,000 loans, however, the AutoML winner scores 0.6625 on 2016 against the production model's 0.6639. With more data, boosting catches up and passes them. AutoML should search on the full training size, or at least confirm its choice there, before its winner replaces a model.

Best TPE parameters: [reports/best_params_tpe.json](reports/best_params_tpe.json). Every trial is in `reports/hpo_tpe_trials.csv` and `reports/hpo_automl_trials.csv`.

### 12.7 Explainability

[explain.py](src/loan_default/explain.py) explains the production model with **TreeSHAP**, which gives exact SHAP values for tree ensembles. It runs on a 3,000-loan sample of 2016, in log-odds of default. Columns derived from one application field (the purpose one-hot columns, the employment missing flag) are summed back into that field. Base value plus contributions reproduces the model's log-odds to within 1e-14.

| Field | Mean abs. SHAP | Direction | Permutation importance (drop in test ROC-AUC) |
|---|---:|---|---:|
| FICO | 0.314 | higher lowers risk | 0.0591 |
| Loan-to-income (derived) | 0.237 | higher raises risk | — |
| DTI | 0.176 | higher raises risk | 0.0232 |
| Purpose | 0.109 | | 0.0135 |
| Housing status | 0.108 | | 0.0144 |
| Loan amount | 0.099 | higher raises risk | 0.0648 |
| State | 0.090 | | 0.0055 |
| Employment length | 0.031 | mixed | 0.0048 |
| Income | 0.027 | higher lowers risk | 0.0255 |

The two methods agree on the main drivers: credit score and how large the loan is relative to the borrower's means. They divide that second effect differently. SHAP credits it to the derived loan-to-income column. Permutation importance shuffles the raw loan amount and income, which also moves loan-to-income, so it credits those raw fields instead. Read together: once loan-to-income is known, income adds little on its own (SHAP 0.027). Partial dependence and ICE curves ([partial_dependence.png](reports/figures/partial_dependence.png)) show default risk falling steadily with FICO and rising with DTI. The individual curves for those two fields run roughly parallel, so their effects hardly depend on the rest of the application. The loan-amount curves fan out instead: a larger loan raises risk sharply for some applicants and barely for others. Income sets the difference, through loan-to-income, which is the model's main interaction. In the beeswarm, a missing employment length (grey) always pushes risk up, which matches the missing-not-at-random finding in section 12.1. The SHAP dependence plot also shows a small upturn at the very lowest DTI: near-zero DTI is slightly riskier than DTI around 8, perhaps because it reflects unreported debt or thin credit files.

**Reason codes.** For a declined applicant, the fields with the largest positive SHAP values are the principal reasons a US lender must state in an adverse action notice (ECOA, Regulation B). For example, one applicant declined at 40.5% predicted default receives *1. Amount requested is large relative to income (loan-to-income 0.43); 2. Credit score (682); 3. Amount requested ($35,000)*. The report walks through a clear approval, an applicant just past the cutoff and a clear rejection. Because state can appear as a reason, compliance should decide whether a geographic reason is acceptable.

Figures: [SHAP importance](reports/figures/shap_importance.png), [beeswarm](reports/figures/shap_beeswarm.png), [dependence](reports/figures/shap_dependence.png), [partial dependence](reports/figures/partial_dependence.png).

## 13. Fairness and ZIP-Code Exclusion

ZIP code is excluded for statistical and ethical reasons.

### Sparse categories

Training contains **879 ZIP-code categories**, with approximately **20%** represented by fewer than 30 loans. Small groups can produce unstable estimates.

### Geographic proxies

Residential geography can act as a proxy for race or ethnicity. Historical redlining illustrates how neighbourhood-based lending restrictions can exclude communities from credit.

We remove ZIP code to reduce dependence on detailed residential location when assessing individual borrowers.

ZIP code would also not have helped accuracy. Embedding it lowers test ROC-AUC by up to 0.003 (section 12.2).

Race and ethnicity labels are unavailable, so racial approval and error disparities cannot be measured directly. ZIP-code exclusion addresses one feature choice; it does not establish fair model outcomes.

### Group audit of the approval policy

[`python -m loan_default.fairness`](src/loan_default/fairness.py) audits the production policy on 2016 ([reports/fairness_report.md](reports/fairness_report.md)). The policy applies the training cutoff for 80% approval unchanged. We audit the groups we can observe, several of which correlate with protected characteristics in US data. The *impact ratio* is a group's approval rate divided by the highest group's; below **0.8** fails the four-fifths rule of thumb used in US disparate-impact analysis. *Approved if repaid* measures equal opportunity: how often borrowers who went on to repay were approved.

| Group | Loans | Default rate | Approved | Impact ratio | Approved if repaid | Predicted − actual |
|---|---:|---:|---:|---:|---:|---:|
| Income $120k and over | 7,006 | 18.4% | 92.5% | 1.00 | 94.2% | −4.3 pp |
| Income $75k–120k | 14,442 | 21.3% | 83.7% | 0.91 | 87.0% | −4.1 pp |
| Income $40k–75k | 21,702 | 24.7% | 75.8% | 0.82 | 80.3% | −4.8 pp |
| **Income under $40k** | 6,850 | 27.9% | 72.3% | **0.78** | 77.2% | −6.6 pp |
| Mortgage | 24,271 | 19.8% | 85.1% | 1.00 | 88.0% | −3.2 pp |
| Own | 6,076 | 23.1% | 82.0% | 0.96 | 84.5% | −4.9 pp |
| Rent | 19,648 | 27.7% | 72.9% | 0.86 | 78.2% | −6.8 pp |
| Employment length given | 46,801 | 22.7% | 81.1% | 1.00 | 84.9% | −4.6 pp |
| **Employment length not given** | 3,199 | 31.3% | 62.7% | **0.77** | 68.6% | −8.3 pp |
| Census regions (4) | 9,127–18,028 | 22.1–24.1% | 78.6–81.7% | ≥ 0.96 | 82.7–85.4% | −4.1 to −5.5 pp |

- **Two groups fail the four-fifths check:** applicants with income under $40k and applicants who gave no employment length. Both groups default more often, so a risk-based policy approves fewer of them. The audit also shows that *repaid* borrowers in these groups are approved less often (77% and 69%, against 94% for the highest income band). That equal-opportunity gap is what a fairness review would question.
- **The calibration error is not shared equally.** Every group's risk is understated in 2016, but the gap is about twice as large for renters (−6.8 pp), low-income applicants (−6.6 pp) and applicants without an employment length (−8.3 pp) as for mortgage holders (−3.2 pp). Recalibrating within groups, or monitoring calibration by group, would matter wherever probabilities are used.
- **Regions are treated alike** (impact ratios 0.96 or higher), so using state does not create a large geographic disparity.

### Mitigations

**1. Remove state.** Test ROC-AUC falls from 0.6639 to 0.6613, and the default rate among the safest 80% rises from 19.48% to 19.57%. The regional approval gap narrows from 3.1 to 1.9 points, while the income and housing gaps barely move. State is a weak proxy for the disparities above, so removing it costs accuracy without fixing them. The production model keeps state, and the result is documented here for compliance review.

**2. Equal-opportunity cutoffs by income band** (post-processing, Hardt et al., 2016). Each band gets its own cutoff, chosen on 2007–2015 out-of-fold scores, so that repaid borrowers in every band are approved at the same rate:

| Policy | Approved | Default rate, approved | Equal-opportunity gap | Lowest impact ratio |
|---|---:|---:|---:|---:|
| Single cutoff (production) | 80.0% | 19.5% | 17.0 pp | 0.78 |
| Cutoff per income band | 78.8% | 19.4% | 3.2 pp | 0.96 |

The cutoffs move from 0.188 for the top band to 0.281 for the bottom band, and the trade-off is small here: approvals fall by 1.2 points and the approved default rate is unchanged. However, a cutoff that depends on a protected characteristic would be disparate treatment under US fair-lending law (ECOA). Income is not protected, but the same mechanism applied to race or sex would be unlawful. This mitigation therefore measures the trade-off; it is not a deployable remedy for protected classes. Lawful alternatives include adding features that explain repayment for thin-file applicants, and a human second look for declined applicants near the cutoff.

## 14. What Worked and What Did Not

### What worked

- **Additional application features:** multivariable models improved substantially on FICO alone.
- **Approval-based evaluation:** matched approval rates connected ranking performance to a lending decision.
- **Later-cohort ranking:** test ROC-AUC remained close to CV performance.
- **Missingness handling:** the pipeline retained employment-information missingness identified in EDA. Section 12.1 confirms that the missing flag matters more than any imputation method.
- **Entity embeddings as boosting inputs:** the best single model (+0.0008 test ROC-AUC), with interpretable state vectors.
- **Causal estimate with refutations:** the effect of a large loan relative to income is stable across four estimators, both periods and three refutation tests.
- **Fairness audit:** it found two groups below the four-fifths threshold and group-specific calibration gaps, neither visible in overall metrics.

### What added little or did not transfer

| Attempt / finding | Result | Lesson |
|---|---|---|
| Default threshold of 0.50 | Very low default recall | Thresholds should reflect the lending objective |
| Hyperparameter search | Largest ROC-AUC gain below 0.001 | Broader tuning produced limited gains |
| Soft voting and stacking | Approximately +0.0006 ROC-AUC; no AP gain | Correlated models added little diversity |
| Loan-to-income | Notebook 03 reports similar boosting performance with and without it | A strong EDA relationship may duplicate existing information |
| Training-period calibration | Underestimated 2016 defaults | Stable ranking does not imply stable probabilities |
| MICE and KNN imputation | Within ±0.0002 test ROC-AUC of median filling | Only one column has gaps, and the fields are weakly related |
| PCA components as model inputs | −0.0016 test ROC-AUC | Rotations break the axis-aligned splits trees rely on |
| Cluster distances as features | +0.0001 | Boosting already finds these interactions |
| Semi-supervised methods | At most +0.004 with 1% of labels; slightly worse with 20% | Outcomes are worth more than unlabeled applications |
| ZIP-code embeddings | −0.003 test ROC-AUC | Sparse categories overfit; excluding ZIP costs nothing |
| Removing state | −0.0026 test ROC-AUC; regional gap 3.1 → 1.9 points; income and housing gaps unchanged | A weak proxy; removing it does not address the main disparities |

The description-availability flag was removed because descriptions became almost absent in later years. Income percentile clipping and FICO bands were not included in the final pipeline.

## 15. Assumptions and Limitations

| Issue | Effect on interpretation | Next step |
|---|---|---|
| Only issued loans are observed | Results may differ for rejected applicants | Obtain broader application data and assess coverage |
| Only resolved outcomes are retained | Cohorts may be selective | Seek fuller follow-up information |
| Outcome dates are unavailable | Time splitting cannot fully reproduce historical label availability | Construct an as-of evaluation |
| Stratified CV mixes years | CV does not simulate future deployment, and hides calibration drift (section 11) | Report year-by-year validation next to CV; tuning on year-ordered folds did not change 2016 results |
| Tuning and policy selection reuse development folds | Selected development scores may be optimistic | Use nested or separate validation in further work |
| Some test information was inspected | Holdout independence is incomplete | Reserve a new cohort for confirmation |
| DTI 999 is treated as missing | Its meaning is undocumented | Verify the code and test alternative treatments |
| Income is self-reported | Reporting errors can affect predictions | Add verification information |
| Few observations have FICO below 660 | Evidence for this group is limited | Obtain more representative observations |
| Probabilities underestimate 2016 defaults | Scores understate later-cohort risk | Assess recalibration on recent labelled data |
| Protected-group labels are absent | Direct racial disparity assessment is unavailable; the audit uses correlated groups (section 13) | Obtain an appropriate subgroup evaluation dataset, or BISG proxies under compliance supervision |
| Interest rate, grade and term are not recorded | The causal estimate (section 12.5) includes the effect of the terms that come with a large loan | Add these fields, or test a loan-to-income cap on a random sample |
| Semi-supervised results simulate missing labels at random | Real unlabeled applicants (rejections) differ systematically from approved ones | Obtain outcomes for a random sample of otherwise-rejected applicants |
| Financial returns and losses are absent | Default reductions cannot be translated directly into profit | Add costs, exposure and recovery information |

## 16. Repository and Reproduction

### Files

| File | Purpose |
|---|---|
| [01_split_and_sample.ipynb](notebooks/01_split_and_sample.ipynb) | Temporal split and sampling |
| [02_eda.ipynb](notebooks/02_eda.ipynb) | Training-data exploration |
| [03_pipeline.ipynb](notebooks/03_pipeline.ipynb) | Pipeline explanation and checks |
| [src/loan_default/preprocessing.py](src/loan_default/preprocessing.py) | Shared preprocessing, used by the notebooks and by training |
| [src/loan_default/policy.py](src/loan_default/policy.py) | Approval policy: approve the lowest-risk share of a batch |
| [src/loan_default/train.py](src/loan_default/train.py) | Trains the final model and saves it with its metadata |
| [src/loan_default/tracking.py](src/loan_default/tracking.py) | Records a training run in MLflow and optionally registers the model |
| [src/loan_default/registry.py](src/loan_default/registry.py) | Promotes a registered model version and exports it for serving |
| [src/loan_default/schema.py](src/loan_default/schema.py) | What a valid application looks like, enforced by the API and by training |
| [src/loan_default/service.py](src/loan_default/service.py) | Prediction API (FastAPI) |
| `Dockerfile`, `.dockerignore` | Two-stage image definition |
| [src/loan_default/drift.py](src/loan_default/drift.py) | Compares the 2016 sample with the training sample; writes `reports/` |
| [src/loan_default/recalibration.py](src/loan_default/recalibration.py) | Tests three recalibration methods across the two halves of 2016; writes `reports/` |
| [src/loan_default/monitor.py](src/loan_default/monitor.py) | Joins outcomes to logged predictions and checks calibration by month |
| [src/loan_default/replay.py](src/loan_default/replay.py) | Replays 2016 through the service log and both monitors; writes `reports/` |
| [src/loan_default/temporal.py](src/loan_default/temporal.py) | Chronological validation, year by year; writes `reports/` |
| [src/loan_default/search.py](src/loan_default/search.py) | Reruns notebook 05's hyperparameter search, on shuffled or year-ordered folds; writes `reports/` |
| [src/loan_default/experiments.py](src/loan_default/experiments.py) | Shared CV-and-2016 comparison harness for the experiments in section 12 |
| [src/loan_default/imputation.py](src/loan_default/imputation.py) | Median, mean, MICE and KNN imputation compared (section 12.1) |
| [src/loan_default/embedding.py](src/loan_default/embedding.py) | Entity-embedding network in NumPy, as classifier and as encoder (section 12.2) |
| [src/loan_default/unsupervised.py](src/loan_default/unsupervised.py) | PCA and k-means segmentation (section 12.3) |
| [src/loan_default/semi_supervised.py](src/loan_default/semi_supervised.py) | Self-training, cluster soft labels, label spreading with few labels (section 12.4) |
| [src/loan_default/causal.py](src/loan_default/causal.py) | IPW and doubly robust (AIPW) effect of loan size relative to income, with refutations (section 12.5) |
| [src/loan_default/hpo.py](src/loan_default/hpo.py) | Optuna TPE with pruning, and AutoML over model families (section 12.6) |
| [src/loan_default/explain.py](src/loan_default/explain.py) | TreeSHAP, partial dependence, reason codes (section 12.7) |
| [src/loan_default/fairness.py](src/loan_default/fairness.py) | Group fairness audit and mitigations (section 13) |
| [docs/model_card.md](docs/model_card.md) | Model card: intended use, metrics, ethical considerations |
| [docs/product_and_data_management.md](docs/product_and_data_management.md) | Users, success metrics, rollout plan, risks, data lineage and governance |
| [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/), [pull_request_template.md](.github/pull_request_template.md) | Issue and pull request templates for planning experiments and reviewing changes |
| [reports/](reports/) | Generated drift, chronological validation, recalibration, search and section 12–13 reports; figures in `reports/figures/` |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | Continuous integration |
| [.github/workflows/drift.yml](.github/workflows/drift.yml) | Weekly drift check on the prediction log |
| [.github/dependabot.yml](.github/dependabot.yml) | Weekly dependency updates |
| [scripts/smoke_test.sh](scripts/smoke_test.sh) | Checks a running service |
| [tests/](tests/) | Unit tests for every module |
| [pyproject.toml](pyproject.toml) | Package definition and ruff, mypy and pytest settings |
| [04_modeling_baseline.ipynb](notebooks/04_modeling_baseline.ipynb) | Baseline, CV, importance and error analysis |
| [05_tuning_and_final_evaluation.ipynb](notebooks/05_tuning_and_final_evaluation.ipynb) | Tuning, ensembles, policies and final evaluation |
| [requirements.txt](requirements.txt) | Exact versions used in CI and the Docker image; Dependabot keeps them current |
| `data/train.csv` | 150,000-loan development sample |
| `data/test.csv` | 50,000-loan test sample |

### Setup

```bash
git clone https://github.com/Doraaadorable/lending_club_default_prediction.git
cd lending_club_default_prediction
python -m pip install -e ".[notebooks]"
cd notebooks
jupyter notebook
```

Scikit-learn 1.4 or later is required for the gradient boosting parameter search in notebook 05.

### Execution

Run notebooks from the `notebooks/` directory in numerical order.

To regenerate the samples, download the raw CSV from Zenodo and place it at:

```text
data/LC_loans_granting_model_dataset.csv
```

The repository already includes `data/train.csv` and `data/test.csv`, so notebooks 02–05 can run without regenerating the split.

Notebook 05 saves intermediate results under `notebooks/results/` and figures under `notebooks/figures/`. Remove the relevant cached results before recomputing after changes to code, data, or parameters.

Sampling, shuffled CV, randomized search, and approval tie-breaking use seed **42**.

### Train the final model

```bash
python -m pip install -e ".[dev]"
python -m loan_default.train
```

This fits the tuned gradient boosting pipeline on `data/train.csv`, evaluates it on `data/test.csv` next to the FICO-only baseline, and writes two files to `models/`:

- `model.joblib`: preprocessing and model as one fitted pipeline. It takes raw application rows, so scoring needs no separate preprocessing step.
- `metadata.json`: hyperparameters, metrics, the approval cutoff derived from out-of-fold training scores, data hashes and the scikit-learn version. Metrics include test ROC-AUC by FICO band and housing status, and how much test ROC-AUC drops when each application field is shuffled.

The run takes about 20 seconds on four cores. It stops without saving if the training or test file breaks the application schema the API enforces (`schema.py`), or if the model does not beat the FICO-only baseline on test ROC-AUC.

The script uses the search winner's exact hyperparameters (section 9) and matches notebook 05's test ROC-AUC (0.6639) and Brier score (0.1706). The approved-loan default rate at 80% approval is 19.48% against the notebook's 19.50%. The FICO-only baseline, which has no tuned settings, moves by the same amount (21.57% against 21.56%), so the gap comes from library versions. Sections 10 and 11 report the notebook's figures.

### Track experiments

```bash
python -m pip install -e ".[tracking]"
python -m loan_default.train --track
python -m loan_default.train --track --max-iter 200 --learning-rate 0.05
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

With `--track`, each run is recorded in MLflow: hyperparameters, every metric in `metadata.json`, the data hashes, and the two saved files. Runs go to a local SQLite database by default; `--tracking-uri` points them at a shared server. The UI compares runs side by side.

### Register and promote a model

```bash
python -m loan_default.train --register
python -m loan_default.registry promote --version 1
python -m loan_default.registry export --out-dir served
```

`--register` records the run and adds the fitted pipeline to the MLflow model registry as a new version of `loan-default`. `promote` points the `champion` alias at a version; `export` writes that version as `model.joblib` and `metadata.json`, the files the service reads. `promote` refuses a version whose test ROC-AUC is below the current champion's; `--force` overrides this for a deliberate rollback. Switching the served model is then a promotion and an export, with no retraining and no image rebuild:

```bash
docker run --rm -p 8000:8000 --read-only -v "$PWD/served:/app/models:ro" loan-default
```

The exported metadata records the registered name, version and alias, so `/metadata` shows which version is live.

### Serve predictions

```bash
python -m pip install -e ".[serve]"
python -m loan_default.train
uvicorn loan_default.service:app --port 8000
```

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "model": ...}`; used by the container health check |
| `GET /metadata` | The training metadata: parameters, metrics, cutoff and data hashes |
| `POST /predict` | The default probability for one application and the decision at the training cutoff |
| `GET /docs` | Interactive API documentation |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "revenue": 60000, "loan_amnt": 12000, "fico_n": 690, "dti_n": 18,
  "emp_length": "5 years", "purpose": "debt_consolidation",
  "home_ownership_n": "RENT", "addr_state": "NY"}'
# {"default_probability":0.1942,"decision":"approve","approval_cutoff":0.2563}
```

Input is validated before it reaches the model: income and amount must be positive, FICO between 300 and 850, the state a two-letter code, and employment length one of the values in the data. Invalid input returns 422 with the reason. `dti_n` and `emp_length` may be omitted and are then treated as missing, as in training. A loan purpose or housing status the model has not seen is scored, not rejected.

The decision applies the cutoff chosen on training data, which approves the safest 80% of training loans. If the model returns a score that is not a probability between 0 and 1, the request fails with 500 and nothing is logged.

**Prediction log.** With `PREDICTION_LOG` set to a file path, every valid prediction is appended as one JSON line: time, the hash of the model file, the application fields, and the score and decision. Send an `application_id` with each request; it is logged but not scored, and it is how outcomes are joined later. Two tools read the log:

```bash
PREDICTION_LOG=logs/predictions.jsonl uvicorn loan_default.service:app --port 8000
python -m loan_default.drift --current logs/predictions.jsonl                          # inputs against training data
python -m loan_default.monitor --log logs/predictions.jsonl --outcomes outcomes.csv     # predicted against actual, by month
```

The drift check needs no outcomes but, as the 2016 replay shows, can miss a model going wrong. The outcome check catches it once outcomes arrive. Both fail with exit status 1, and [drift.yml](.github/workflows/drift.yml) runs them weekly.

**Docker**

```bash
docker build -t loan-default .
docker run --rm -p 8000:8000 --read-only --cap-drop ALL loan-default
scripts/smoke_test.sh
```

The build has two stages. The first trains the model from `data/`; the second installs the package and copies in only the trained model, so the served image contains no loan data. Versions are pinned in `requirements.txt`, and the container runs as a non-root user.

The container's filesystem is read-only, so a prediction log needs a writable mount:

```bash
docker run --rm -p 8000:8000 --read-only --tmpfs /app/logs -e PREDICTION_LOG=/app/logs/predictions.jsonl loan-default
```

### Code checks

```bash
python -m pytest        # unit tests
ruff check .            # lint: unused code, naming, missing type hints and docstrings
ruff format --check .   # formatting
mypy                    # static type check (strict)
```

GitHub Actions runs these on every push to `main` and every pull request ([ci.yml](.github/workflows/ci.yml)). When they pass, two more jobs run:

- **image** builds the Docker image, starts it read-only as a non-root user, and runs the smoke test. The smoke test sends 200 predictions and fails if the 95th-percentile latency exceeds 0.2 seconds; the job then fails if the container uses more than 600 MiB.
- **analysis** reruns the year-by-year validation, the 2016 drift report, the 2016 replay, the fairness audit and the SHAP explanations, and attaches the reports to the run.

Two more automations:

- [drift.yml](.github/workflows/drift.yml) runs every Monday against the service's prediction log, downloaded from the repository variable `PREDICTION_LOG_URL`. It fails if any input shifts by PSI 0.10 or more, or, when `OUTCOMES_URL` is also set, if any month's default rate is more than 2 points from the prediction. GitHub notifies the repository owner of failed scheduled runs.
- [Dependabot](.github/dependabot.yml) opens a pull request each week when a pinned package or a GitHub Action has an update; CI then tests it.

## 17. Production Readiness

Scored with the ML Test Score rubric ([Breck et al., 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)): 28 tests in four groups. A test scores 1 if it runs automatically, in CI or on every training run; 0.5 if it was done and documented but is run by hand; 0 if it is not done. The overall score is the lowest group total, because strength in one group does not make up for a gap in another.

**Features and data: 4.5**

| Test | Score | Evidence |
|---|---:|---|
| Feature expectations are written as a schema | 1 | `schema.py`, enforced on every API request and on the training and test files at every training run |
| Every feature earns its place | 1 | Every training run records each field's permutation importance on test data |
| No feature costs more than it adds | 0.5 | ZIP code, title and description dropped with reasons (section 5) |
| Features meet policy requirements | 0.5 | ZIP code excluded as a geographic proxy; state's effect on group disparities measured (section 13) |
| The data pipeline has privacy controls | 0 | Public data; none built |
| New features can be added quickly | 0.5 | One cleaning function and one column list, both under test |
| Feature code is tested | 1 | Unit tests for cleaning, in CI |

**Model development: 5**

| Test | Score | Evidence |
|---|---:|---|
| The model spec is reviewed and versioned | 1 | In git, merged through pull requests, checked in CI |
| Offline metrics track online impact | 0 | No live outcomes |
| Hyperparameters are tuned | 0.5 | 40-configuration search, rerun with `loan_default.search` |
| The cost of staleness is known | 1 | CI reruns the year-by-year validation on every push |
| A simpler model is not better | 1 | Training fails unless the model beats the FICO-only baseline |
| Quality holds on important slices | 1 | Every training run records test ROC-AUC by FICO band and housing status |
| The model is checked for inclusion | 0.5 | CI audits approval and error rates by income, housing, region and employment information (section 13); protected-class labels are absent |

**ML infrastructure: 5**

| Test | Score | Evidence |
|---|---:|---|
| Training is reproducible | 1 | A test retrains and compares the full metadata, in CI |
| Model code is unit tested | 1 | Training, policy, search and analysis modules, in CI |
| The whole pipeline is integration tested | 1 | CI builds the image, which trains the model, then smoke-tests the running service |
| Quality is checked before serving | 1 | Training stops if the model loses to the baseline; promotion stops if it loses to the champion |
| A single prediction can be traced | 0.5 | Logs tie each prediction to a model hash; stepping through the pipeline is manual |
| New models go through a canary | 0 | Not built |
| A served model can be rolled back | 0.5 | Promote an earlier registry version with `--force` and re-export (section 16) |

**Monitoring: 5.5**

| Test | Score | Evidence |
|---|---:|---|
| Upstream changes trigger a notification | 0.5 | Dependabot reports package and Action updates; the data source is a fixed file |
| Data invariants hold in training and serving | 1 | The same schema is enforced on API requests and on training data |
| Training and serving compute the same features | 1 | One fitted pipeline does both; a test checks the exported model scores identically |
| The served model is not too stale | 0.5 | A weekly drift job is in place; there is no live prediction log for it to read yet |
| Outputs are numerically valid | 1 | The service rejects any score outside 0–1, with a test |
| Serving speed and memory hold steady | 1 | CI fails on p95 latency above 0.2 s or memory above 600 MiB |
| Prediction quality holds | 0.5 | `loan_default.monitor` joins outcomes by application id and fails on a monthly gap above 2 points; CI runs it on the 2016 replay, and there are no live outcomes yet |

**Overall: 4.5**, up from 2. Features and data now set it. The remaining zeros need things this project does not have: live outcomes, protected-group labels, a deployment to canary, and private data to protect.

## 18. Further Work

| Step | What it would add | What it needs |
|---|---|---|
| A later untouched cohort | Confirm the drift, calibration and recalibration findings on loans not used for any decision here | The raw dataset (section 16), and a cohort with enough follow-up that its outcomes are not selected toward early defaults (section 3) |
| Reason codes in the API | Return each decision's top SHAP reasons from `/predict` | Check that the `shap` dependency fits the 600 MiB memory and 0.2 s latency budgets |
| Calibration monitoring by group | Catch the larger understatement for renters and low-income applicants (section 13) | Group labels in the prediction log and the outcome join |
| Embeddings in production | Use the best single model (section 12.2) | A confirming cohort, since the gain is within one cohort's noise |

## 19. Conclusion

Application-time features improve risk ranking beyond FICO alone.

At **80% approval**, tuned gradient boosting retained 40,000 test loans with a **19.50%** default rate, compared with **21.56%** for FICO. This represents **827 fewer defaulted loans** at the same approval volume.

The main remaining issue is calibration: ranking transferred to 2016, but probabilities underestimated its higher default level. The inputs were stable; loans of the same predicted risk defaulted more often. Chronological validation shows the underestimation building from 2014, a trend that shuffled CV hid. Recalibrating on recent loans removes about half of the gap, and the rest remains because the level kept rising. Tuning on year-ordered folds did not change this. Replaying 2016 as if the model were live, input monitoring stays green all year; only a check against outcomes catches the error, and only as fast as outcomes arrive. Further work should confirm these findings on a later cohort.

The further techniques in section 12 confirm that the production pipeline is close to what this data supports. Model-based imputation, PCA, clustering, semi-supervised learning and AutoML each moved test ROC-AUC by less than 0.002; only entity embeddings gave a consistent small gain. Their value lies elsewhere. The causal analysis turns a correlation into a policy estimate: a loan of at least a quarter of annual income raises default by about 9 points, and by 11 points below FICO 680. SHAP turns scores into reason codes that a declined applicant can be given. The fairness audit finds what overall metrics hide: low-income applicants and applicants without an employment length fall below the four-fifths threshold, and the model understates risk most for exactly these groups.

## References

- Ariza-Garzón, M. J., Sanz-Guerrero, M., Arroyo Gallardo, J., & Lending Club. (2024). *Lending Club loan dataset for granting models* (Version 0.1) [Dataset]. Universidad Complutense de Madrid. <https://doi.org/10.5281/zenodo.11295916>

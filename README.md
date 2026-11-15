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
| Drift, year-by-year validation, recalibration | Section 11 |
| Training, tracking, registry, API, Docker, CI | Section 15 |
| Production readiness (ML Test Score) | Section 16 |

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

## 12. Fairness and ZIP-Code Exclusion

ZIP code is excluded for statistical and ethical reasons.

### Sparse categories

Training contains **879 ZIP-code categories**, with approximately **20%** represented by fewer than 30 loans. Small groups can produce unstable estimates.

### Geographic proxies

Residential geography can act as a proxy for race or ethnicity. Historical redlining illustrates how neighbourhood-based lending restrictions can exclude communities from credit.

We remove ZIP code to reduce dependence on detailed residential location when assessing individual borrowers.

State remains through smoothed target encoding. A useful extension is to compare model performance and approval patterns with and without state.

Race and ethnicity labels are unavailable, so racial approval and error disparities cannot be measured directly. ZIP-code exclusion addresses one feature choice; it does not establish fair model outcomes.

## 13. What Worked and What Did Not

### What worked

- **Additional application features:** multivariable models improved substantially on FICO alone.
- **Approval-based evaluation:** matched approval rates connected ranking performance to a lending decision.
- **Later-cohort ranking:** test ROC-AUC remained close to CV performance.
- **Missingness handling:** the pipeline retained employment-information missingness identified in EDA.

### What added little or did not transfer

| Attempt / finding | Result | Lesson |
|---|---|---|
| Default threshold of 0.50 | Very low default recall | Thresholds should reflect the lending objective |
| Hyperparameter search | Largest ROC-AUC gain below 0.001 | Broader tuning produced limited gains |
| Soft voting and stacking | Approximately +0.0006 ROC-AUC; no AP gain | Correlated models added little diversity |
| Loan-to-income | Notebook 03 reports similar boosting performance with and without it | A strong EDA relationship may duplicate existing information |
| Training-period calibration | Underestimated 2016 defaults | Stable ranking does not imply stable probabilities |

The description-availability flag was removed because descriptions became almost absent in later years. Income percentile clipping and FICO bands were not included in the final pipeline.

## 14. Assumptions and Limitations

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
| Protected-group labels are absent | Direct racial disparity assessment is unavailable | Obtain an appropriate subgroup evaluation dataset |
| Financial returns and losses are absent | Default reductions cannot be translated directly into profit | Add costs, exposure and recovery information |

## 15. Repository and Reproduction

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
| [src/loan_default/temporal.py](src/loan_default/temporal.py) | Chronological validation, year by year; writes `reports/` |
| [src/loan_default/search.py](src/loan_default/search.py) | Reruns notebook 05's hyperparameter search, on shuffled or year-ordered folds; writes `reports/` |
| [reports/](reports/) | Generated drift, chronological validation, recalibration and search results |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | Continuous integration |
| [.github/workflows/drift.yml](.github/workflows/drift.yml) | Weekly drift check on the prediction log |
| [.github/dependabot.yml](.github/dependabot.yml) | Weekly dependency updates |
| [scripts/smoke_test.sh](scripts/smoke_test.sh) | Checks a running service |
| [tests/](tests/) | Unit tests for preprocessing, the policy and training |
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

**Prediction log.** With `PREDICTION_LOG` set to a file path, every valid prediction is appended as one JSON line: time, the hash of the model file, the application fields, and the score and decision. The drift tool compares a log with the training data:

```bash
PREDICTION_LOG=logs/predictions.jsonl uvicorn loan_default.service:app --port 8000
python -m loan_default.drift --current logs/predictions.jsonl    # writes reports/log_drift_report.md
```

Logged applications have no outcomes yet, so this compares inputs only. Section 11 shows the limit: in 2016 the inputs were stable while default rates rose.

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
- **analysis** reruns the year-by-year validation and the 2016 drift report and attaches both to the run.

Two more automations:

- [drift.yml](.github/workflows/drift.yml) runs every Monday against the service's prediction log, downloaded from the repository variable `PREDICTION_LOG_URL`, and fails if any input shifts by PSI 0.10 or more. GitHub notifies the repository owner of failed scheduled runs.
- [Dependabot](.github/dependabot.yml) opens a pull request each week when a pinned package or a GitHub Action has an update; CI then tests it.

## 16. Production Readiness

Scored with the ML Test Score rubric ([Breck et al., 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)): 28 tests in four groups. A test scores 1 if it runs automatically, in CI or on every training run; 0.5 if it was done and documented but is run by hand; 0 if it is not done. The overall score is the lowest group total, because strength in one group does not make up for a gap in another.

**Features and data: 4.5**

| Test | Score | Evidence |
|---|---:|---|
| Feature expectations are written as a schema | 1 | `schema.py`, enforced on every API request and on the training and test files at every training run |
| Every feature earns its place | 1 | Every training run records each field's permutation importance on test data |
| No feature costs more than it adds | 0.5 | ZIP code, title and description dropped with reasons (section 5) |
| Features meet policy requirements | 0.5 | ZIP code excluded as a geographic proxy (section 12) |
| The data pipeline has privacy controls | 0 | Public data; none built |
| New features can be added quickly | 0.5 | One cleaning function and one column list, both under test |
| Feature code is tested | 1 | Unit tests for cleaning, in CI |

**Model development: 4.5**

| Test | Score | Evidence |
|---|---:|---|
| The model spec is reviewed and versioned | 1 | In git, merged through pull requests, checked in CI |
| Offline metrics track online impact | 0 | No live outcomes |
| Hyperparameters are tuned | 0.5 | 40-configuration search, rerun with `loan_default.search` |
| The cost of staleness is known | 1 | CI reruns the year-by-year validation on every push |
| A simpler model is not better | 1 | Training fails unless the model beats the FICO-only baseline |
| Quality holds on important slices | 1 | Every training run records test ROC-AUC by FICO band and housing status |
| The model is checked for inclusion | 0 | No protected-group labels (section 12) |

**ML infrastructure: 5**

| Test | Score | Evidence |
|---|---:|---|
| Training is reproducible | 1 | A test retrains and compares the full metadata, in CI |
| Model code is unit tested | 1 | Training, policy, search and analysis modules, in CI |
| The whole pipeline is integration tested | 1 | CI builds the image, which trains the model, then smoke-tests the running service |
| Quality is checked before serving | 1 | Training stops if the model loses to the baseline; promotion stops if it loses to the champion |
| A single prediction can be traced | 0.5 | Logs tie each prediction to a model hash; stepping through the pipeline is manual |
| New models go through a canary | 0 | Not built |
| A served model can be rolled back | 0.5 | Promote an earlier registry version with `--force` and re-export (section 15) |

**Monitoring: 5**

| Test | Score | Evidence |
|---|---:|---|
| Upstream changes trigger a notification | 0.5 | Dependabot reports package and Action updates; the data source is a fixed file |
| Data invariants hold in training and serving | 1 | The same schema is enforced on API requests and on training data |
| Training and serving compute the same features | 1 | One fitted pipeline does both; a test checks the exported model scores identically |
| The served model is not too stale | 0.5 | A weekly drift job is in place; there is no live prediction log for it to read yet |
| Outputs are numerically valid | 1 | The service rejects any score outside 0–1, with a test |
| Serving speed and memory hold steady | 1 | CI fails on p95 latency above 0.2 s or memory above 600 MiB |
| Prediction quality holds | 0 | Predictions are logged, but outcomes are not joined back |

**Overall: 4.5**, up from 2. Features and model development now set it. The remaining zeros need things this project does not have: live outcomes, protected-group labels, a deployment to canary, and private data to protect.

## 17. Further Work

| Step | What it would add | What it needs |
|---|---|---|
| A later untouched cohort | Confirm the drift, calibration and recalibration findings on loans not used for any decision here | The raw dataset (section 15), and a cohort with enough follow-up that its outcomes are not selected toward early defaults (section 3) |

## 18. Conclusion

Application-time features improve risk ranking beyond FICO alone.

At **80% approval**, tuned gradient boosting retained 40,000 test loans with a **19.50%** default rate, compared with **21.56%** for FICO. This represents **827 fewer defaulted loans** at the same approval volume.

The main remaining issue is calibration: ranking transferred to 2016, but probabilities underestimated its higher default level. The inputs were stable; loans of the same predicted risk defaulted more often. Chronological validation shows the underestimation building from 2014, a trend that shuffled CV hid. Recalibrating on recent loans removes about half of the gap, and the rest remains because the level kept rising. Tuning on year-ordered folds did not change this. Further work should confirm these findings on a later cohort and assess dependence on geographic features.

## References

- Ariza-Garzón, M. J., Sanz-Guerrero, M., Arroyo Gallardo, J., & Lending Club. (2024). *Lending Club loan dataset for granting models* (Version 0.1) [Dataset]. Universidad Complutense de Madrid. <https://doi.org/10.5281/zenodo.11295916>

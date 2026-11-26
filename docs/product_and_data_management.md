# Product and data management

This document covers the product side of the project: who the model is for, how success is measured, how it would be rolled out, and how the data behind it is managed. The modelling details are in the [README](../README.md); the model's intended use and limits are in the [model card](model_card.md).

## 1. Problem statement

Lending Club approves or rejects personal-loan applications. Approving too freely raises defaults; approving too strictly turns away borrowers who would have repaid and lowers lending volume. Today the simplest screen is the applicant's FICO score.

**Product goal:** at the same approval volume, approve a set of loans that defaults less often than a FICO-only screen, and give each decision an explanation that an underwriter and a declined applicant can understand.

## 2. Users and their needs

| User | Need | What the project provides |
|---|---|---|
| Credit-policy team | Choose an approval rate and see its default rate | Approval curves and the 80% policy (README sections 7-10) |
| Underwriters | Know why a score is high | Per-applicant SHAP contributions (`loan_default.explain`) |
| Compliance and fair-lending officers | Evidence that the policy does not disadvantage groups without justification; adverse action reasons | Fairness audit (`loan_default.fairness`), ZIP-code exclusion, model card; adverse action reasons are not yet built |
| Model risk management (validation) | Reproducible training, documented assumptions, monitoring | Data hashes in metadata, MLflow runs, CI, drift and outcome monitors, ML Test Score (README section 17) |
| Engineering / MLOps | A service that is safe to deploy and roll back | FastAPI service, Docker image, MLflow registry with champion alias, smoke and latency tests |
| Applicants | A fast decision and, if declined, the main reasons | Sub-0.2 s p95 scoring; decline reasons are future work (README section 18) |

## 3. Success metrics

Model metrics are only useful if they move a business metric, so the two are tracked together.

| Level | Metric | Target / guardrail | Where measured |
|---|---|---|---|
| Business | Default rate among approved loans at a fixed approval rate | Below the FICO-only screen at the same rate (19.5% vs 21.6% at 80% on 2016) | `train.py` metadata, README section 10 |
| Business | Defaults avoided at the same volume | 827 fewer defaults per 40,000 approvals vs FICO | README section 10 |
| Model | ROC-AUC, average precision | Training stops unless test ROC-AUC beats the FICO baseline; AP is reported, not gated | `train.py` |
| Model | Calibration gap (mean predicted minus actual) | Within 2 points per month, or the monitor alerts | `monitor.py` |
| Fairness | Adverse impact ratio by group | Ratios below 0.8 (a screening threshold, not a legal test) trigger a review | `fairness.py` |
| Operations | p95 latency, memory | Below 0.2 s and 600 MiB, or CI fails | `ci.yml` |
| Operations | Input drift (PSI) | Below 0.10 per feature, or the weekly job fails | `drift.yml` |

## 4. Scope

**In scope (this release):** scoring applications at the time of application, an approval rule at a chosen approval rate, explanations, monitoring, and offline analyses of further modeling techniques (README section 12).

**Out of scope:** interest-rate pricing, loan term choice, profit optimization (returns and recoveries are not in the data), collections, and any use of the score outside the application decision.

## 5. Rollout plan

| Stage | What happens | Exit criterion |
|---|---|---|
| 1. Shadow | The service scores live applications; underwriters decide as today; every score is logged | Two months of logs; drift job green; latency within budget |
| 2. Champion / challenger | A share of applications (e.g. 10%) is decided by the model; the rest by the current process | Approved-loan default rate (on early-payment signals) no worse than the current process; any group below the 0.8 screening ratio reviewed and explained |
| 3. Full use with human review | The model decides; applications near the cutoff go to an underwriter | Monthly calibration gap within 2 points; quarterly fairness audit passes |
| Rollback | `registry promote --version <previous> --force`, then `export` | Any guardrail breached |

Outcomes arrive slowly (a default can take many months), so stage 2 needs an early signal such as missed first payments; the replay in README section 11 shows how much the outcome delay limits detection.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Probabilities run low in a later cohort (seen in 2016) | Outcome monitor by month; recalibration tested (README section 11) |
| Bias against groups correlated with protected classes | ZIP code excluded; fairness audit with a 0.8 screening threshold; state-removal comparison; legal review before launch |
| Model learns only from approved loans (selection bias) | Documented; the semi-supervised experiment (labels hidden at random) tests methods for unlabeled loans but not reject inference, which needs rejected-application data |
| Confusing correlation with cause in policy decisions | Causal analysis of loan size relative to income, with refutation tests and stated assumptions |
| Silent breakage of the service | Schema validation, CI smoke test, numeric output check, health endpoint |

## 7. Data management

### Source and licence

Ariza-Garzón et al. (2024), Lending Club loan dataset for granting models, Zenodo, CC BY 4.0. The raw file (167 MB) is not stored in git; the two samples derived from it are.

### Lineage

```
LC_loans_granting_model_dataset.csv (Zenodo, 1,347,681 loans)
  └─ notebook 01: keep 2007-2016, split by issue year, stratified sample with seed 42
       ├─ data/train.csv  150,000 loans, 2007-2015
       └─ data/test.csv    50,000 loans, 2016
            └─ preprocessing.clean()  (no learned values)
                 └─ fitted pipeline in models/model.joblib, with SHA-256 of both files in metadata.json
```

### Versioning and reproducibility

- Every training run writes the SHA-256 of the training and test files to `metadata.json`, and MLflow stores them with the run, so a model can be traced to the exact data it saw.
- Code is versioned in git and changes go through pull requests with CI.
- Random seeds are fixed (42) for sampling, CV, search, tie-breaking and every experiment.

### Data quality checks

| Check | Where |
|---|---|
| Schema (types, ranges, allowed values) on training data, test data and every API request | `schema.py`, run by `train.py` and `service.py` |
| Train and test loan ids do not overlap | `train.py` |
| Placeholder values (DTI = 999) treated as missing; extreme DTI capped | `preprocessing.clean` |
| Distribution shift of every input against training data | `drift.py`, weekly in `drift.yml` |

### Data dictionary

The variable table in README section 4 is the data dictionary: meaning, treatment and the reason for each excluded column.

### Privacy and retention

The dataset is public and contains no names or full addresses; ZIP codes are already truncated to three digits. In a live deployment the prediction log would hold application data, so it should be access-controlled, retained only as long as model monitoring and regulation require, and kept out of the serving image (the image already contains no loan data).

### Governance

| Artifact | Owner role | Review |
|---|---|---|
| Training data samples | Data owner | On any regeneration: hashes change and are recorded |
| Model version in the registry | Model owner | Promotion requires beating the champion's test ROC-AUC |
| Fairness audit | Compliance | Each promotion and quarterly |
| Model card | Model owner | Each promotion |

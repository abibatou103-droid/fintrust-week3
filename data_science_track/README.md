# FinTrust Data Science Track — Week 3

This project upgrades the Week 2 FinTrust risk-review model into a more robust, leakage-aware, and validation-focused solution for the Data Science track.

## Objective

Build on the Week 2 baseline and move from a simple proof of concept to a more defensible predictive risk model for FinTrust transactions. The modelling goal remains the same: estimate the probability that a transaction should be flagged for risk review using information available at the time the transaction is processed.

The target remains synthetic and educational only, and must not be presented as a real fraud decision.

## What was improved from Week 2

- Added a stricter time-based holdout split instead of a random split.
- Kept leakage-prone variables out of the modelling pipeline (`Transaction_Status`, `Account_Status`, `Customer_Name`, raw IDs).
- Added more context-aware engineered features such as transaction-hour, time-since-last-transaction, customer-level amount deviations, and night/weekend indicators.
- Trained and compared multiple classification models.
- Reviewed false positives and false negatives by transaction type and channel.
- Saved a candidate final model and the model comparison table for Week 4 integration work.

## Candidate final model

The selected model is Logistic Regression with class balancing and a temporally ordered train/test split.

Why this model was selected:
- best balance of recall and ROC-AUC among the models tested;
- highly interpretable via coefficients;
- less likely to overfit than a pure tree ensemble on this small, synthetic dataset;
- more suitable for a first-stage risk review tool that needs explanatory signals.

## Validation strategy

The data were split by time to reflect real deployment conditions:
- Train period: earliest 80% of transactions by date
- Test period: final 20% of transactions by date

This is more realistic than a random split and avoids optimistic results caused by temporal leakage.

## Model comparison

Results on the time-based holdout test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.588 | 0.274 | 0.693 | 0.393 | 0.681 | 0.338 |
| Random Forest | 0.770 | 0.375 | 0.294 | 0.330 | 0.680 | 0.332 |
| HistGradientBoosting | 0.800 | 0.393 | 0.071 | 0.121 | 0.654 | 0.306 |

The logistic model offers the best trade-off between recall and explanatory clarity, which matters for a risk-review use case. A Random Forest achieved a better accuracy score because it favoured the majority class, but it missed too many true positives. The final choice therefore prioritises sensitivity and business usefulness instead of raw accuracy alone.

## Error analysis summary

The false positive and false negative patterns were reviewed by transaction type and channel. This showed that the biggest prediction errors concentrated around transaction patterns that are noisy or weakly aligned with the synthetic target, which is consistent with the label being educational and not a real fraud signal.

The project saves the full error summary in `outputs/error_analysis_summary.csv`.

## Feature refinement decisions

Key decisions applied during this upgrade:
- Kept transaction amount but transformed it using `log1p` to reduce skew.
- Included contextual features such as amount relative to the customer median and mean, transaction hour, night flag, weekend indicator, and time since the previous transaction.
- Excluded post-transaction and post-review signals such as `Transaction_Status` and `Account_Status`.
- Used customer-level context available before transaction approval, rather than future information.

## Deliverables produced

- `fintrust_week3_ds_pipeline.py`: end-to-end model training pipeline
- `prepared_modelling_dataset.csv`: assembled modelling dataset
- `candidate_model.pkl`: selected model artefact
- `outputs/model_comparison.csv`: model comparison table
- `outputs/final_model_summary.json`: summary of the chosen model and validation setup
- `outputs/confusion_matrix_*.png`: confusion matrices for each model
- `outputs/logistic_regression_coefficients.csv`: coefficient-based interpretation
- `outputs/error_analysis_summary.csv`: summary of false positive / false negative patterns

## Limitations

1. The target is synthetic and educational. It should not be used as a real fraud or compliance signal.
2. The label may not fully reflect the real risk drivers in the bank setting.
3. This is a transaction-level classification task with a moderate class imbalance; therefore, model performance must be interpreted with care.
4. Additional tuning, calibration, and out-of-time validation would still be required before operational deployment.

## Next step for Week 4

The candidate model should be moved into a final validation and documentation phase, with deeper calibration and stakeholder-facing explanation of the risk drivers and business implications.

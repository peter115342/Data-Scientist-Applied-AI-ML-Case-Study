# Commercial Auto Risk Scoring

## Approach

The solution trains a logistic-regression pipeline to estimate each commercial-auto policy line's probability of at least one claim. It applies median imputation with missingness indicators and scaling to numeric variables, and most-frequent imputation with one-hot encoding to categorical variables. Skewed count and monetary variables also receive `log1p` transformations.

The pipeline validates its input schema, removes duplicate training records, treats negative premiums as missing, safely maps unknown business types to `other`, validates prediction output, and saves the fitted model and metadata. It also checks numeric and categorical input drift between training and scoring data.

## Key assumptions

- The model predicts claim occurrence, not claim severity, expected loss, or a recommended premium.
- Only underwriting-time information is used. Current-period claim outcomes, post-outcome fields, late-payment behaviour, and the train-only `has_safety_program` field are excluded.
- One record represents one policy line, identified by `policy_id`.

## Target definition

`target = claim_count > 0`: a policy has at least one claim in its current policy period. The final `risk_score` is the predicted probability of this target.

## Validation approach

The data is split by time to reflect the future scoring use case: 2020 is used for model selection, 2021 for validation, and 2022 is held out for the final test. Six logistic-regression regularization strengths are compared using 2021 ROC-AUC; the selected value is `C=0.01`. The final pipeline is then refit on all 2020-2022 data before scoring 2023-2024 policies.

Performance is reported with ROC-AUC, average precision, Brier score, observed claim rate, and separate AL/APD metrics. A temporal learning curve is displayed when the scorer runs.

## Run the solution

1. In Databricks, open **Workspace**, choose a location, then select **Create > Git folder**.
2. Choose **GitHub** and use:

   ```text
   https://github.com/peter115342/Data-Scientist-Applied-AI-ML-Case-Study.git
   ```

3. Select the `master` branch and create the Git folder.
4. Attach serverless compute using **Environment version 5**.
5. Open `eda.py` and choose **Run all**, then open `risk_scorer.py` and choose **Run all**.

The first notebook cells read dependencies from `pyproject.toml` and install them with `%uv pip`. `eda.py` displays the exploratory analysis; `risk_scorer.py` writes `predictions.csv` and displays the temporal learning curve.

## Limitations and trade-offs

- This is a claim-occurrence model, not a severity or expected-loss model.
- The final performance estimate uses one held-out future year (2022), so it is not a production approval.
- Logistic regression is intentionally interpretable and robust, but more flexible models could be evaluated after establishing governance and avoiding overfitting.
- Production deployment would still require governed ingestion, scheduled scoring, model registry and approvals, monitoring once outcomes arrive, fairness analysis, human review thresholds, and business-value measurement.

# Databricks notebook source
# MAGIC %pip install -q "polars" "numpy" "scikit-learn"

# COMMAND ----------

if "dbutils" in globals():
    globals()["dbutils"].library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Commercial auto policy risk scoring
# MAGIC
# MAGIC Fits a model on historical data and scores the current period.
# MAGIC
# MAGIC - `train.csv`: policies from 2020–2022, including claim outcomes
# MAGIC - `score.csv`: policies from 2023–2024, without outcomes

# COMMAND ----------


import json
from pathlib import Path

from joblib import dump
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

# ---- configuration ----
TRAIN_DATA_PATH = "train.csv"
SCORE_DATA_PATH = "score.csv"
PREDICTIONS_PATH = "predictions.csv"
ARTIFACT_DIR = Path("artifacts")
VALIDATION_CUTOFF = "2021-01-01"
TEST_CUTOFF = "2022-01-01"
CANDIDATE_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MODEL_CONFIG = {"max_iter": 1000}
LOG_FEATURES = {
    "vehicle_count": "log_vehicle_count",
    "driver_count": "log_driver_count",
    "prior_year_mileage_000": "log_prior_year_mileage",
    "prior_loss_amount": "log_prior_loss_amount",
    "annual_premium": "log_annual_premium",
}

# COMMAND ----------

# ---- load ----
df2 = pl.read_csv(TRAIN_DATA_PATH)
df3 = pl.read_csv(SCORE_DATA_PATH)

print("train:", df2.shape)
print("score:", df3.shape)

# COMMAND ----------

# ---- preprocessing ----

df2 = df2.with_columns(pl.col("business_type").str.strip_chars().str.to_lowercase())
df3 = df3.with_columns(pl.col("business_type").str.strip_chars().str.to_lowercase())
df2 = df2.with_columns(pl.col("snapshot_date").str.slice(5, 2).alias("snapshot_month"))
df3 = df3.with_columns(pl.col("snapshot_date").str.slice(5, 2).alias("snapshot_month"))

# Add log features to capture diminishing effects in skewed variables.
log_feature_expressions = [
    pl.col(source).clip(lower_bound=0).log1p().alias(feature)
    for source, feature in LOG_FEATURES.items()
]
df2 = df2.with_columns(log_feature_expressions)
df3 = df3.with_columns(log_feature_expressions)

# COMMAND ----------

# ---- target ----
# using binary had-a-claim flag as target
df2 = df2.with_columns((pl.col("claim_count") > 0).cast(pl.Int64).alias("target"))
print("positive rate:", round(df2["target"].mean(), 3))

# COMMAND ----------

# ---- features ----
category_columns = [
    "coverage_type",
    "business_type",
    "state",
    "snapshot_month",
    "payment_frequency",
]
numeric_columns = [
    "vehicle_count",
    "vehicle_avg_age",
    "driver_count",
    "driver_avg_age",
    "years_in_business",
    "prior_year_mileage_000",
    "prior_apd_claim_count",
    "prior_al_claim_count",
    "prior_loss_amount",
    "deductible",
    "coverage_limit_000",
    "annual_premium",
    "risk_score_external",
    "num_heavy_vehicles",
    *LOG_FEATURES.values(),
]
feat_cols = category_columns + numeric_columns

missing_score_features = set(feat_cols) - set(df3.columns)
if missing_score_features:
    raise ValueError(f"score.csv is missing required features: {missing_score_features}")
if df3["policy_id"].null_count() or df3["policy_id"].n_unique() != df3.height:
    raise ValueError("score.csv must contain one non-null, unique policy_id per row")
if df2["policy_id"].null_count():
    raise ValueError("train.csv must not contain null policy_id values")

training_record_columns = ["snapshot_date", "target", *feat_cols]
conflicting_training_policies = (
    df2.group_by("policy_id")
    .agg(pl.struct(training_record_columns).n_unique().alias("record_count"))
    .filter(pl.col("record_count") > 1)
)
if conflicting_training_policies.height:
    raise ValueError("train.csv contains policy_id values with conflicting model records")

# Keep one normalized training record per policy line.
training_rows_before_deduplication = df2.height
df2 = df2.unique(subset=["policy_id"], keep="first", maintain_order=True)
deduplicated_training_rows = training_rows_before_deduplication - df2.height
print("deduplicated training rows:", deduplicated_training_rows)

X = df2[feat_cols]
y = df2["target"]

# COMMAND ----------

# ---- fit model ----

# Use 2020 for model selection, 2021 for validation, and leave 2022 untouched
# for the final estimate of performance on unseen data.
tuning_train_df = df2.filter(pl.col("snapshot_date") < VALIDATION_CUTOFF)
validation_df = df2.filter(
    (pl.col("snapshot_date") >= VALIDATION_CUTOFF)
    & (pl.col("snapshot_date") < TEST_CUTOFF)
)
train_df = df2.filter(pl.col("snapshot_date") < TEST_CUTOFF)
test_df = df2.filter(pl.col("snapshot_date") >= TEST_CUTOFF)

X_tuning_train = tuning_train_df[feat_cols]
y_tuning_train = tuning_train_df["target"]
X_validation = validation_df[feat_cols]
y_validation = validation_df["target"]
X_train = train_df[feat_cols]
X_test = test_df[feat_cols]
y_train = train_df["target"]
y_test = test_df["target"]


def build_model(model_config):
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                numeric_columns,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                category_columns,
            ),
        ]
    )
    return make_pipeline(preprocessor, LogisticRegression(**model_config))


model_selection_results = []
for candidate_c in CANDIDATE_C_VALUES:
    candidate_config = {**MODEL_CONFIG, "C": candidate_c}
    candidate_model = build_model(candidate_config)
    candidate_model.fit(X_tuning_train, y_tuning_train)
    candidate_probabilities = candidate_model.predict_proba(X_validation)[:, 1]
    model_selection_results.append(
        {
            "C": candidate_c,
            "validation_roc_auc": roc_auc_score(
                y_validation, candidate_probabilities
            ),
            "validation_average_precision": average_precision_score(
                y_validation, candidate_probabilities
            ),
            "validation_brier_score": brier_score_loss(
                y_validation, candidate_probabilities
            ),
        }
    )

selected_result = max(
    model_selection_results, key=lambda result: result["validation_roc_auc"]
)
selected_model_config = {**MODEL_CONFIG, "C": selected_result["C"]}
print("selected model config:", selected_model_config)

tmp = build_model(selected_model_config)
tmp.fit(X_train, y_train)

# COMMAND ----------

# ---- evaluate ----
x1 = tmp.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, x1)
test_average_precision = average_precision_score(y_test, x1)
test_brier_score = brier_score_loss(y_test, x1)
test_mean_predicted_risk = x1.mean()
test_observed_claim_rate = y_test.mean()
print("test_roc_auc:", test_roc_auc)
print("test_average_precision:", test_average_precision)
print("test_brier_score:", test_brier_score)
print("test_mean_predicted_risk:", test_mean_predicted_risk)
print("test_observed_claim_rate:", test_observed_claim_rate)
# results look solid

# COMMAND ----------

# ---- refit final model ----
tmp.fit(X, y)

ARTIFACT_DIR.mkdir(exist_ok=True)
dump(tmp, ARTIFACT_DIR / "risk_model.joblib")
(ARTIFACT_DIR / "model_metadata.json").write_text(
    json.dumps(
        {
            "model_type": "logistic_regression",
            "training_period": "2020-2022",
            "training_rows_before_deduplication": training_rows_before_deduplication,
            "deduplicated_training_rows": deduplicated_training_rows,
            "model_selection_training_period": "2020",
            "validation_period": "2021",
            "test_period": "2022",
            "test_roc_auc": test_roc_auc,
            "test_average_precision": test_average_precision,
            "test_brier_score": test_brier_score,
            "test_mean_predicted_risk": test_mean_predicted_risk,
            "test_observed_claim_rate": test_observed_claim_rate,
            "feature_columns": feat_cols,
            "log1p_features": LOG_FEATURES,
            "validation_cutoff": VALIDATION_CUTOFF,
            "test_cutoff": TEST_CUTOFF,
            "model_selection_metric": "roc_auc",
            "model_selection_results": model_selection_results,
            "model_config": selected_model_config,
        },
        indent=2,
    ),
    encoding="utf-8",
)

# COMMAND ----------

# ---- score new policies ----
X2 = df3[feat_cols]
preds = tmp.predict_proba(X2)[:, 1]
df3 = df3.with_columns(pl.Series("risk_score", preds))
predictions = df3.select(["policy_id", "risk_score"])
if predictions.height != df3.height or predictions["risk_score"].null_count():
    raise ValueError("predictions must contain one non-null risk_score per policy")
predictions.write_csv(PREDICTIONS_PATH)
print(f"done -- {PREDICTIONS_PATH} written")

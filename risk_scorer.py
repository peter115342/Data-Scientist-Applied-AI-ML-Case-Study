# Databricks notebook source
# MAGIC %python
# MAGIC import tomllib
# MAGIC from pathlib import Path
# MAGIC
# MAGIC requirements_path = Path("/tmp/commercial-auto-risk-requirements.txt")
# MAGIC dependencies = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
# MAGIC requirements_path.write_text("\n".join(dependencies), encoding="utf-8")

# COMMAND ----------

# MAGIC %uv pip install -r /tmp/commercial-auto-risk-requirements.txt

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
from matplotlib import pyplot as plt
import numpy as np
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
LEARNING_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
DRIFT_BIN_COUNT = 10
DRIFT_PSI_ALERT_THRESHOLD = 0.25
KNOWN_BUSINESS_TYPES = {"trucking", "retail", "construction", "service", "other"}
MODEL_CONFIG = {"max_iter": 1000}
LOG_FEATURES = {
    "vehicle_count": "log_vehicle_count",
    "driver_count": "log_driver_count",
    "prior_year_mileage_000": "log_prior_year_mileage",
    "prior_loss_amount": "log_prior_loss_amount",
    "annual_premium": "log_annual_premium",
}
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
RAW_FEATURE_COLUMNS = (
    set(feat_cols) - {"snapshot_month", *LOG_FEATURES.values()}
)
REQUIRED_TRAIN_COLUMNS = {"policy_id", "snapshot_date", "claim_count", *RAW_FEATURE_COLUMNS}
REQUIRED_SCORE_COLUMNS = {"policy_id", "snapshot_date", *RAW_FEATURE_COLUMNS}

# COMMAND ----------

# ---- load ----
df2 = pl.read_csv(TRAIN_DATA_PATH)
df3 = pl.read_csv(SCORE_DATA_PATH)


def validate_input_schema(frame, dataset_name, required_columns):
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing_columns)}"
        )

    parsed_snapshot_dates = frame.select(
        pl.col("snapshot_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )
    if parsed_snapshot_dates["snapshot_date"].null_count():
        raise ValueError(f"{dataset_name} contains null or invalid snapshot_date values")


def normalize_business_types(frame):
    normalized_business_type = pl.col("business_type").str.strip_chars().str.to_lowercase()
    unknown_business_types = (
        frame.select(normalized_business_type.alias("business_type"))
        .filter(
            pl.col("business_type").is_not_null()
            & ~pl.col("business_type").is_in(KNOWN_BUSINESS_TYPES)
        )
        .group_by("business_type")
        .len()
        .sort("business_type")
        .to_dicts()
    )
    normalized_frame = frame.with_columns(
        pl.when(normalized_business_type.is_null())
        .then(None)
        .when(normalized_business_type.is_in(KNOWN_BUSINESS_TYPES))
        .then(normalized_business_type)
        .otherwise(pl.lit("other"))
        .alias("business_type")
    )
    return normalized_frame, unknown_business_types


def numeric_population_stability_index(reference, comparison):
    reference = np.asarray(reference.drop_nulls().to_list(), dtype=float)
    comparison = np.asarray(comparison.drop_nulls().to_list(), dtype=float)
    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]
    if not len(reference) or not len(comparison):
        return None

    bin_edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, DRIFT_BIN_COUNT + 1))
    )
    if len(bin_edges) < 3:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    reference_share = np.histogram(reference, bins=bin_edges)[0] / len(reference)
    comparison_share = np.histogram(comparison, bins=bin_edges)[0] / len(comparison)
    reference_share = np.clip(reference_share, 1e-6, None)
    comparison_share = np.clip(comparison_share, 1e-6, None)
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def build_feature_drift_report(reference_frame, scoring_frame):
    numeric_drift = []
    for column in numeric_columns:
        psi = numeric_population_stability_index(
            reference_frame[column], scoring_frame[column]
        )
        numeric_drift.append(
            {
                "feature": column,
                "population_stability_index": psi,
                "alert": psi is not None and psi >= DRIFT_PSI_ALERT_THRESHOLD,
                "training_missing_rate": reference_frame[column].null_count()
                / reference_frame.height,
                "scoring_missing_rate": scoring_frame[column].null_count()
                / scoring_frame.height,
            }
        )

    categorical_drift = []
    for column in category_columns:
        reference_values = reference_frame[column].fill_null("<missing>").to_list()
        scoring_values = scoring_frame[column].fill_null("<missing>").to_list()
        categories = set(reference_values) | set(scoring_values)
        distribution_distance = 0.5 * sum(
            abs(
                reference_values.count(category) / len(reference_values)
                - scoring_values.count(category) / len(scoring_values)
            )
            for category in categories
        )
        categorical_drift.append(
            {
                "feature": column,
                "total_variation_distance": distribution_distance,
                "new_scoring_categories": sorted(
                    set(scoring_values) - set(reference_values)
                ),
            }
        )

    return {"numeric": numeric_drift, "categorical": categorical_drift}


validate_input_schema(df2, "train.csv", REQUIRED_TRAIN_COLUMNS)
validate_input_schema(df3, "score.csv", REQUIRED_SCORE_COLUMNS)

print("train:", df2.shape)
print("score:", df3.shape)

# COMMAND ----------

# ---- preprocessing ----

df2, unknown_training_business_types = normalize_business_types(df2)
df3, unknown_scoring_business_types = normalize_business_types(df3)
print("unknown scoring business types mapped to other:", unknown_scoring_business_types)
df2 = df2.with_columns(pl.col("snapshot_date").str.slice(5, 2).alias("snapshot_month"))
df3 = df3.with_columns(pl.col("snapshot_date").str.slice(5, 2).alias("snapshot_month"))

invalid_training_premiums = df2.filter(pl.col("annual_premium") < 0).height
invalid_scoring_premiums = df3.filter(pl.col("annual_premium") < 0).height
# Treat negative premiums as missing.
premium_sanitization = (
    pl.when(pl.col("annual_premium") < 0).then(None).otherwise(pl.col("annual_premium"))
)
df2 = df2.with_columns(premium_sanitization.alias("annual_premium"))
df3 = df3.with_columns(premium_sanitization.alias("annual_premium"))
print("invalid training premiums converted to null:", invalid_training_premiums)
print("invalid scoring premiums converted to null:", invalid_scoring_premiums)

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

missing_score_features = set(feat_cols) - set(df3.columns)
if missing_score_features:
    raise ValueError(
        f"score.csv is missing required features: {missing_score_features}"
    )
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
    raise ValueError(
        "train.csv contains policy_id values with conflicting model records"
    )

# Keep one normalized training record per policy line.
training_rows_before_deduplication = df2.height
df2 = df2.unique(subset=["policy_id"], keep="first", maintain_order=True)
deduplicated_training_rows = training_rows_before_deduplication - df2.height
print("deduplicated training rows:", deduplicated_training_rows)

feature_drift_report = build_feature_drift_report(df2, df3)
numeric_drift_alerts = [
    result["feature"]
    for result in feature_drift_report["numeric"]
    if result["alert"]
]
print("numeric PSI alerts:", numeric_drift_alerts)

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


def plot_temporal_learning_curve(results):
    training_rows = [result["training_rows"] for result in results]
    training_scores = [result["training_roc_auc"] for result in results]
    validation_scores = [result["validation_roc_auc"] for result in results]

    plt.figure(figsize=(8, 4.5))
    plt.plot(training_rows, training_scores, marker="o", label="Training (2020)")
    plt.plot(training_rows, validation_scores, marker="o", label="Validation (2021)")
    plt.xlabel("Training rows")
    plt.ylabel("ROC-AUC")
    plt.title("Temporal learning curve")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()


def calculate_coverage_metrics(frame, probabilities):
    evaluation_frame = frame.select(["coverage_type", "target"]).with_columns(
        pl.Series("risk_score", probabilities),
        pl.col("coverage_type").fill_null("<missing>"),
    )
    coverage_metrics = []
    for coverage_type in sorted(evaluation_frame["coverage_type"].unique().to_list()):
        segment = evaluation_frame.filter(pl.col("coverage_type") == coverage_type)
        target = segment["target"]
        risk_score = segment["risk_score"]
        has_both_outcomes = target.n_unique() == 2
        coverage_metrics.append(
            {
                "coverage_type": coverage_type,
                "policy_count": segment.height,
                "claim_rate": float(target.mean()),
                "mean_predicted_risk": float(risk_score.mean()),
                "roc_auc": float(roc_auc_score(target, risk_score))
                if has_both_outcomes
                else None,
                "average_precision": float(average_precision_score(target, risk_score))
                if has_both_outcomes
                else None,
                "brier_score": float(brier_score_loss(target, risk_score)),
            }
        )
    return coverage_metrics


model_selection_results = []
for candidate_c in CANDIDATE_C_VALUES:
    candidate_config = {**MODEL_CONFIG, "C": candidate_c}
    candidate_model = build_model(candidate_config)
    candidate_model.fit(X_tuning_train, y_tuning_train)
    candidate_probabilities = candidate_model.predict_proba(X_validation)[:, 1]
    model_selection_results.append(
        {
            "C": candidate_c,
            "validation_roc_auc": roc_auc_score(y_validation, candidate_probabilities),
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

learning_curve_results = []
ordered_tuning_train = tuning_train_df.sort("snapshot_date")
for fraction in LEARNING_CURVE_FRACTIONS:
    row_count = round(ordered_tuning_train.height * fraction)
    learning_frame = ordered_tuning_train.head(row_count)
    learning_model = build_model(selected_model_config)
    learning_model.fit(learning_frame[feat_cols], learning_frame["target"])
    learning_curve_results.append(
        {
            "training_rows": row_count,
            "training_roc_auc": roc_auc_score(
                learning_frame["target"],
                learning_model.predict_proba(learning_frame[feat_cols])[:, 1],
            ),
            "validation_roc_auc": roc_auc_score(
                y_validation,
                learning_model.predict_proba(X_validation)[:, 1],
            ),
        }
    )

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
test_coverage_metrics = calculate_coverage_metrics(test_df, x1)
print("test_roc_auc:", test_roc_auc)
print("test_average_precision:", test_average_precision)
print("test_brier_score:", test_brier_score)
print("test_mean_predicted_risk:", test_mean_predicted_risk)
print("test_observed_claim_rate:", test_observed_claim_rate)
print("test_coverage_metrics:", test_coverage_metrics)
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
            "invalid_training_premiums": invalid_training_premiums,
            "invalid_scoring_premiums": invalid_scoring_premiums,
            "unknown_training_business_types": unknown_training_business_types,
            "unknown_scoring_business_types": unknown_scoring_business_types,
            "model_selection_training_period": "2020",
            "validation_period": "2021",
            "test_period": "2022",
            "test_roc_auc": test_roc_auc,
            "test_average_precision": test_average_precision,
            "test_brier_score": test_brier_score,
            "test_mean_predicted_risk": test_mean_predicted_risk,
            "test_observed_claim_rate": test_observed_claim_rate,
            "test_coverage_metrics": test_coverage_metrics,
            "feature_columns": feat_cols,
            "log1p_features": LOG_FEATURES,
            "validation_cutoff": VALIDATION_CUTOFF,
            "test_cutoff": TEST_CUTOFF,
            "model_selection_metric": "roc_auc",
            "model_selection_results": model_selection_results,
            "temporal_learning_curve": learning_curve_results,
            "scoring_feature_drift": feature_drift_report,
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
plot_temporal_learning_curve(learning_curve_results)
plt.show()

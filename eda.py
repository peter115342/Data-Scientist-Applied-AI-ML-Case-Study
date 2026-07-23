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
# MAGIC # Commercial auto risk scoring - exploratory data analysis
# MAGIC

# COMMAND ----------

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

plt.style.use("seaborn-v0_8-whitegrid")
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
FIGURES_DIR = Path("artifacts/figures")
FIGURE_DPI = 150


def save_and_show(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"saved figure: {figure_path}")
    plt.show()

# COMMAND ----------

# ---- load ----
train = pl.read_csv("train.csv")
score = pl.read_csv("score.csv")
train = train.with_columns((pl.col("claim_count") > 0).cast(pl.Int8).alias("target"))

print("train shape:", train.shape)
print("score shape:", score.shape)
print(
    "train snapshot range:",
    train["snapshot_date"].min(),
    "to",
    train["snapshot_date"].max(),
)
print(
    "score snapshot range:",
    score["snapshot_date"].min(),
    "to",
    score["snapshot_date"].max(),
)
print("overall claim rate:", round(train["target"].mean(), 4))

# COMMAND ----------

# ---- target rate over time ----
yearly_claim_rate = (
    train.with_columns(pl.col("snapshot_date").str.slice(0, 4).alias("year"))
    .group_by("year")
    .agg(pl.len().alias("policies"), pl.mean("target").alias("claim_rate"))
    .sort("year")
)
print(yearly_claim_rate)

plt.figure(figsize=(7, 4))
plt.plot(yearly_claim_rate["year"], yearly_claim_rate["claim_rate"] * 100, marker="o")
plt.title("Claim rate by snapshot year")
plt.xlabel("Snapshot year")
plt.ylabel("Claim rate (%)")
plt.ylim(bottom=0)
save_and_show("eda-yearly-claim-rate.png")

# COMMAND ----------

# ---- claim rates by coverage and business type ----
coverage_claim_rate = (
    train.group_by("coverage_type")
    .agg(pl.len().alias("policies"), pl.mean("target").alias("claim_rate"))
    .sort("coverage_type")
)
business_claim_rate = (
    train.with_columns(pl.col("business_type").str.strip_chars().str.to_lowercase())
    .group_by("business_type")
    .agg(pl.len().alias("policies"), pl.mean("target").alias("claim_rate"))
    .filter(pl.col("policies") >= 50)
    .sort("claim_rate", descending=True)
)

print("claim rate by coverage")
print(coverage_claim_rate)
print("claim rate by business type")
print(business_claim_rate)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(
    coverage_claim_rate["coverage_type"], coverage_claim_rate["claim_rate"] * 100
)
axes[0].set_title("Claim rate by coverage")
axes[0].set_ylabel("Claim rate (%)")

axes[1].bar(
    business_claim_rate["business_type"], business_claim_rate["claim_rate"] * 100
)
axes[1].set_title("Claim rate by business type")
axes[1].set_ylabel("Claim rate (%)")
axes[1].tick_params(axis="x", rotation=30)
save_and_show("eda-claim-rates.png")

# COMMAND ----------

# ---- external risk score relationship ----
risk_band = (
    pl.when(pl.col("risk_score_external") < 20)
    .then(pl.lit("0-19"))
    .when(pl.col("risk_score_external") < 40)
    .then(pl.lit("20-39"))
    .when(pl.col("risk_score_external") < 60)
    .then(pl.lit("40-59"))
    .when(pl.col("risk_score_external") < 80)
    .then(pl.lit("60-79"))
    .otherwise(pl.lit("80-100"))
)
risk_band_claim_rate = (
    train.filter(pl.col("risk_score_external").is_not_null())
    .with_columns(risk_band.alias("risk_band"))
    .group_by("risk_band")
    .agg(pl.len().alias("policies"), pl.mean("target").alias("claim_rate"))
    .sort("risk_band")
)
print(risk_band_claim_rate)

plt.figure(figsize=(8, 4))
plt.bar(risk_band_claim_rate["risk_band"], risk_band_claim_rate["claim_rate"] * 100)
plt.title("Claim rate by external risk-score band")
plt.xlabel("External risk score")
plt.ylabel("Claim rate (%)")
save_and_show("eda-risk-score-bands.png")

# COMMAND ----------

# ---- numeric relationships with the claim target ----
numeric_features = [
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
]
correlations = (
    pl.DataFrame(
        {
            "feature": numeric_features,
            "correlation": [
                train.select(
                    pl.corr(pl.col(feature).fill_null(0), pl.col("target"))
                ).item()
                for feature in numeric_features
            ],
        }
    )
    .with_columns(pl.col("correlation").abs().alias("absolute_correlation"))
    .sort("absolute_correlation", descending=True)
)
print(correlations)

plt.figure(figsize=(8, 5))
plt.barh(correlations["feature"], correlations["correlation"])
plt.title("Linear correlation with claim indicator")
plt.xlabel("Correlation")
plt.gca().invert_yaxis()
save_and_show("eda-numeric-correlations.png")

# COMMAND ----------

# ---- train versus score population comparison ----
plt.figure(figsize=(8, 4))
plt.hist(
    train["risk_score_external"].drop_nulls().to_numpy(),
    bins=25,
    alpha=0.6,
    density=True,
    label="train (2020-2022)",
)
plt.hist(
    score["risk_score_external"].drop_nulls().to_numpy(),
    bins=25,
    alpha=0.6,
    density=True,
    label="score (2023-2024)",
)
plt.title("External risk-score distribution: train versus score")
plt.xlabel("External risk score")
plt.ylabel("Density")
plt.legend()
save_and_show("eda-train-score-risk-distribution.png")

# COMMAND ----------

# ---- data quality and point-in-time availability ----
post_outcome_columns = [
    "first_claim_reported_date",
    "claim_paid_amount_current_period",
    "claim_status_current_period",
    "days_to_first_claim_report",
]
availability = pl.DataFrame(
    {
        "column": post_outcome_columns,
        "train_null_rate": [
            train[column].null_count() / train.height for column in post_outcome_columns
        ],
        "score_null_rate": [
            score[column].null_count() / score.height for column in post_outcome_columns
        ],
    }
)
print("Post-outcome fields: inspect only, do not use them as underwriting features")
print(availability)

missingness = pl.DataFrame(
    {
        "column": train.columns,
        "train_null_rate": [
            train[column].null_count() / train.height for column in train.columns
        ],
        "score_null_rate": [
            score[column].null_count() / score.height
            if column in score.columns
            else None
            for column in train.columns
        ],
    }
).filter((pl.col("train_null_rate") > 0) | (pl.col("score_null_rate") > 0))
print("Columns with missing values")
print(missingness.sort("train_null_rate", descending=True))

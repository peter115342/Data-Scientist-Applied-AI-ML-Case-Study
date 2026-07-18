# Databricks notebook source
# MAGIC %pip install -q "polars" "numpy" "scikit-learn"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Commercial auto policy risk scoring
# MAGIC
# MAGIC Fits a model on historical data and scores the current period.
# MAGIC
# MAGIC - `train.csv`: policies from 2020–2022, including claim outcomes
# MAGIC - `score.csv`: policies from 2023–2024, without outcomes

# COMMAND ----------


import polars as pl
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

# ---- load ----
df2 = pl.read_csv("train.csv")
df3 = pl.read_csv("score.csv")

print("train:", df2.shape)
print("score:", df3.shape)

# COMMAND ----------

# ---- preprocessing ----

df2 = df2.with_columns(pl.col("business_type").str.strip_chars().str.to_lowercase())
df3 = df3.with_columns(pl.col("business_type").str.strip_chars().str.to_lowercase())
df2 = df2.with_columns(pl.col("snapshot_date").str.slice(5, 2).alias("snapshot_month"))
df3 = df3.with_columns(pl.col("snapshot_date").str.slice(5, 2).alias("snapshot_month"))

# COMMAND ----------

# ---- target ----
# using binary had-a-claim flag as target
df2 = df2.with_columns((pl.col("claim_count") > 0).cast(pl.Int64).alias("target"))
print("positive rate:", round(df2["target"].mean(), 3))

# COMMAND ----------

# ---- features ----
category_columns = ["coverage_type", "business_type", "state", "snapshot_month"]
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
]
feat_cols = category_columns + numeric_columns

missing_score_features = set(feat_cols) - set(df3.columns)
if missing_score_features:
    raise ValueError(f"score.csv is missing required features: {missing_score_features}")
if df3["policy_id"].null_count() or df3["policy_id"].n_unique() != df3.height:
    raise ValueError("score.csv must contain one non-null, unique policy_id per row")

X = df2[feat_cols]
y = df2["target"]

# COMMAND ----------

# ---- fit model ----

# time-based split - train on 2020-2021, test on 2022 policies
train_df = df2.filter(pl.col("snapshot_date") < "2022-01-01")
test_df = df2.filter(pl.col("snapshot_date") >= "2022-01-01")
X_train = train_df[feat_cols]
X_test = test_df[feat_cols]
y_train = train_df["target"]
y_test = test_df["target"]

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
        ("categorical", OneHotEncoder(handle_unknown="ignore"), category_columns),
    ]
)
tmp = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
tmp.fit(X_train, y_train)

# COMMAND ----------

# ---- evaluate ----
x1 = tmp.predict_proba(X_test)[:, 1]
print("roc_auc:", roc_auc_score(y_test, x1))
print("average_precision:", average_precision_score(y_test, x1))
# results look solid

# COMMAND ----------

# ---- refit final model ----
tmp.fit(X, y)

# COMMAND ----------

# ---- score new policies ----
X2 = df3[feat_cols]
preds = tmp.predict_proba(X2)[:, 1]
df3 = df3.with_columns(pl.Series("risk_score", preds))
predictions = df3.select(["policy_id", "risk_score"])
if predictions.height != df3.height or predictions["risk_score"].null_count():
    raise ValueError("predictions must contain one non-null risk_score per policy")
predictions.write_csv("predictions.csv")
print("done -- predictions.csv written")

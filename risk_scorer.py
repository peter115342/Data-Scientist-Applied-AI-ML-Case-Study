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
import polars.selectors as cs
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COMMAND ----------

# ---- load ----
df2 = pl.read_csv("train.csv")
df3 = pl.read_csv("score.csv")

print("train:", df2.shape)
print("score:", df3.shape)

# COMMAND ----------

# ---- preprocessing ----

# fill numeric missing with 0 -- good enough for now
df2 = df2.with_columns(cs.numeric().fill_null(0))
df3 = df3.with_columns(cs.numeric().fill_null(0))

# encode categoricals for train
category_columns = ["coverage_type", "business_type", "state"]
category_maps = {
    column: {
        value: code
        for code, value in enumerate(sorted(df2[column].drop_nulls().unique().to_list()))
    }
    for column in category_columns
}
df2 = df2.with_columns(
    [
        pl.col(column)
        .replace_strict(category_maps[column], default=-1, return_dtype=pl.Int64)
        .alias(column)
        for column in category_columns
    ]
)

# same for score
df3 = df3.with_columns(
    [
        pl.col(column)
        .replace_strict(category_maps[column], default=-1, return_dtype=pl.Int64)
        .alias(column)
        for column in category_columns
    ]
)

# COMMAND ----------

# ---- target ----
# using binary had-a-claim flag as target
df2 = df2.with_columns((pl.col("claim_count") > 0).cast(pl.Int64).alias("target"))
print("positive rate:", round(df2["target"].mean(), 3))

# COMMAND ----------

# ---- features ----
feat_cols = [
    "coverage_type",
    "vehicle_count",
    "vehicle_avg_age",
    "driver_count",
    "driver_avg_age",
    "years_in_business",
    "prior_year_mileage_000",
    "business_type",
    "state",
    "prior_apd_claim_count",
    "prior_al_claim_count",
    "prior_loss_amount",
    "deductible",
    "coverage_limit_000",
    "annual_premium",
    "risk_score_external",
    "num_heavy_vehicles",
]

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

tmp = LogisticRegression(max_iter=1000)
tmp.fit(X_train, y_train)

# COMMAND ----------

# ---- evaluate ----
x1 = tmp.predict(X_test)
print("accuracy:", accuracy_score(y_test, x1))
# results look solid

# COMMAND ----------

# ---- score new policies ----
X2 = df3[feat_cols]
preds = tmp.predict_probability(X2)[:, 1]

df3 = df3.with_columns(pl.Series("risk_score", preds))
df3.select(["policy_id", "risk_score"]).write_csv("predictions.csv")
print("done -- predictions.csv written")

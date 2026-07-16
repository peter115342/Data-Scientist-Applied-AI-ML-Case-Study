# Databricks notebook source
# MAGIC %pip install -r ./requirements.txt

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

from pathlib import Path

import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COMMAND ----------

# ---- load ----
DATA_DIR = Path.cwd()
df2 = pl.read_csv(DATA_DIR / "train.csv")
df3 = pl.read_csv(DATA_DIR / "score.csv")

print("train:", df2.shape)
print("score:", df3.shape)

# COMMAND ----------

# ---- preprocessing ----

# fill numeric missing with 0 -- good enough for now
df2[df2.select_dtypes("number").columns] = df2.select_dtypes("number").fillna(0)
df3[df3.select_dtypes("number").columns] = df3.select_dtypes("number").fillna(0)

# encode categoricals for train
df2["coverage_type"] = df2["coverage_type"].astype("category").cat.codes
df2["business_type"] = df2["business_type"].astype("category").cat.codes
df2["state"] = df2["state"].astype("category").cat.codes

# same for score
df3["coverage_type"] = df3["coverage_type"].astype("category").cat.codes
df3["business_type"] = df3["business_type"].astype("category").cat.codes
df3["state"] = df3["state"].astype("category").cat.codes

# COMMAND ----------

# ---- target ----
# using binary had-a-claim flag as target
df2["target"] = (df2["claim_count"] > 0).astype(int)
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
    "late_payment_count",
    "claim_paid_amount_current_period",
    "days_to_first_claim_report",
]

X = df2[feat_cols]
y = df2["target"]

# COMMAND ----------

# ---- fit model ----

# random split -- not time-based
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tmp = LogisticRegression(max_iter=1000)
tmp.fit(X_train, y_train)

# COMMAND ----------

# ---- evaluate ----
x1 = tmp.predict(X_train)
print("accuracy:", accuracy_score(y_train, x1))
# results look solid

# COMMAND ----------

# ---- score new policies ----
X2 = df3[feat_cols]
preds = tmp.predict(X2)

df3["risk_score"] = preds
df3[["policy_id", "risk_score"]].to_csv("predictions.csv")
print("done -- predictions.csv written")

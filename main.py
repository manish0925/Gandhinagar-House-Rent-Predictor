import pandas as pd
import os
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

Model_file = "model.pkl"
Pipeline_file = "pipeline.pkl"


# ------------------ PIPELINE ------------------
def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # ✅ FIX
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline


# ------------------ TRAIN ------------------
if not os.path.exists(Model_file):

    df = pd.read_csv("gandhinagar_rent_dataset.csv")

    # 🔥 CLEAN DATA (IMPORTANT FIX)
    df = df.drop(columns=["Unnamed: 7", "id"], errors="ignore")

    # Convert numeric columns safely
    df["full_rent"] = pd.to_numeric(df["full_rent"], errors="coerce")

    # Drop rows where target missing
    df = df.dropna(subset=["full_rent"])

    # ------------------ STRATIFIED SPLIT ------------------
    df["rent_cat"] = pd.cut(
        df["full_rent"],
        bins=[0, 12000, 18000, 25000, 32000, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(df, df["rent_cat"]):
        strat_train_set = df.loc[train_index].drop("rent_cat", axis=1)
        strat_test_set = df.loc[test_index].drop("rent_cat", axis=1)

    strat_test_set.to_csv("gandhinagar_test_set.csv", index=False)

    # ------------------ TRAIN DATA ------------------
    rent_data = strat_train_set.copy()

    target = "full_rent"
    rent_labels = rent_data[target]

    # ❗ AREA DROP ONLY IF EXISTS
    drop_cols = [target]
    if "area" in rent_data.columns:
        drop_cols.append("area")

    rent_predictors = rent_data.drop(drop_cols, axis=1)

    # ------------------ TYPE FIX ------------------
    # Convert numeric-like columns
    for col in rent_predictors.columns:
        rent_predictors[col] = pd.to_numeric(rent_predictors[col], errors="ignore")

    # ------------------ SPLIT FEATURES ------------------
    num_attribs = rent_predictors.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = rent_predictors.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Numerical Columns:", num_attribs)
    print("Categorical Columns:", cat_attribs)

    # ------------------ PIPELINE ------------------
    pipeline = build_pipeline(num_attribs, cat_attribs)

    rent_prepared = pipeline.fit_transform(rent_predictors)

    # ------------------ MODEL ------------------
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(rent_prepared, rent_labels)

    joblib.dump(model, Model_file)
    joblib.dump(pipeline, Pipeline_file)

    print("✅ Model and Pipeline saved successfully")

# ------------------ PREDICTION ------------------
else:
    print("📦 Loading saved model...")

    model = joblib.load(Model_file)
    pipeline = joblib.load(Pipeline_file)

    input_data = pd.read_csv("gandhinagar_test_set.csv")

    # Same preprocessing
    input_data_clean = input_data.copy()

    if "full_rent" in input_data_clean.columns:
        input_data_clean = input_data_clean.drop("full_rent", axis=1)

    if "area" in input_data_clean.columns:
        input_data_clean = input_data_clean.drop("area", axis=1)

    transform_data = pipeline.transform(input_data_clean)

    predictions = model.predict(transform_data)

    input_data["Predicted_Rent"] = predictions

    input_data.to_csv("gandhinagar_prediction.csv", index=False)

    print("✅ Prediction file saved successfully")
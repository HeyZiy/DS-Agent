import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def _parse_schedule_time(series):
    raw = series.astype(str).str.replace(".0", "", regex=False).str.strip()
    raw = raw.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_datetime(raw, format="%Y%m%d%H%M", errors="coerce")


def _engineer_features(df):
    df = df.copy()

    schedule_columns = ["S_DEPTIME", "S_ARRTIME", "P_DEPTIME", "P_ARRTIME", "R_DEPTIME"]
    for col in schedule_columns:
        df[col] = _parse_schedule_time(df[col])

    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], errors="coerce")

    df["flight_prefix"] = df["FLIGHTID"].astype(str).str.extract(r"([A-Za-z]+)", expand=False)
    df["flight_number"] = pd.to_numeric(
        df["FLIGHTID"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    for col in schedule_columns + ["entry_timestamp"]:
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_minute"] = df[col].dt.minute
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_day"] = df[col].dt.day

    df["scheduled_block_minutes"] = (
        (df["S_ARRTIME"] - df["S_DEPTIME"]).dt.total_seconds() / 60.0
    )
    df["planned_block_minutes"] = (
        (df["P_ARRTIME"] - df["P_DEPTIME"]).dt.total_seconds() / 60.0
    )
    df["real_departure_delay_minutes"] = (
        (df["R_DEPTIME"] - df["P_DEPTIME"]).dt.total_seconds() / 60.0
    )
    df["entry_since_scheduled_departure_minutes"] = (
        (df["entry_timestamp"] - df["S_DEPTIME"]).dt.total_seconds() / 60.0
    )
    df["entry_since_planned_departure_minutes"] = (
        (df["entry_timestamp"] - df["P_DEPTIME"]).dt.total_seconds() / 60.0
    )
    df["entry_since_real_departure_minutes"] = (
        (df["entry_timestamp"] - df["R_DEPTIME"]).dt.total_seconds() / 60.0
    )
    df["scheduled_arrival_minus_entry_minutes"] = (
        (df["S_ARRTIME"] - df["entry_timestamp"]).dt.total_seconds() / 60.0
    )
    df["planned_arrival_minus_entry_minutes"] = (
        (df["P_ARRTIME"] - df["entry_timestamp"]).dt.total_seconds() / 60.0
    )
    df["schedule_plan_arrival_gap_minutes"] = (
        (df["S_ARRTIME"] - df["P_ARRTIME"]).dt.total_seconds() / 60.0
    )
    df["schedule_plan_departure_gap_minutes"] = (
        (df["S_DEPTIME"] - df["P_DEPTIME"]).dt.total_seconds() / 60.0
    )
    datetime_cols = schedule_columns + ["entry_timestamp"]
    df = df.drop(columns=datetime_cols)

    return df


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numeric", numeric_pipeline, numeric_cols),
        ],
        remainder="drop",
    )


class BlendedRegressor:
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.asarray(weights, dtype=float) / np.sum(weights)

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return predictions @ self.weights


def train_model(X_train, y_train, X_valid, y_valid):
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    gradient_boosting = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_iter=400,
        random_state=SEED,
    )
    extra_trees = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )

    gradient_boosting.fit(X_train_processed, y_train)
    extra_trees.fit(X_train_processed, y_train)

    gb_pred = gradient_boosting.predict(X_valid_processed)
    et_pred = extra_trees.predict(X_valid_processed)

    gb_rmse = compute_metrics_for_regression(y_valid, gb_pred)
    et_rmse = compute_metrics_for_regression(y_valid, et_pred)

    blend_candidates = [
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    ]
    best_models = [gradient_boosting]
    best_weights = [1.0]
    best_rmse = gb_rmse

    if et_rmse < best_rmse:
        best_models = [extra_trees]
        best_weights = [1.0]
        best_rmse = et_rmse

    for weights in blend_candidates:
        blended = weights[0] * gb_pred + weights[1] * et_pred
        rmse = compute_metrics_for_regression(y_valid, blended)
        if rmse < best_rmse:
            best_models = [gradient_boosting, extra_trees]
            best_weights = list(weights)
            best_rmse = rmse

    print(f"HistGradientBoosting RMSE: {gb_rmse:.4f}")
    print(f"ExtraTrees RMSE: {et_rmse:.4f}")
    print(f"Selected validation RMSE: {best_rmse:.4f}")

    return {
        "feature_columns": X_train.columns.tolist(),
        "preprocessor": preprocessor,
        "model": BlendedRegressor(best_models, best_weights),
    }


def predict(model_bundle, X):
    X = X.copy()
    X = X.reindex(columns=model_bundle["feature_columns"])
    X_processed = model_bundle["preprocessor"].transform(X)
    y_pred = model_bundle["model"].predict(X_processed)
    return np.clip(y_pred, 0, None)


if __name__ == "__main__":
    data_df = pd.read_csv("train.csv")
    data_df = data_df.dropna(subset=["Time_of_Entry"])
    data_df = _engineer_features(data_df)

    X = data_df.drop(columns=["Time_of_Entry"])
    y = data_df["Time_of_Entry"].to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
    )

    model_bundle = train_model(X_train, y_train, X_valid, y_valid)

    y_valid_pred = predict(model_bundle, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"Final RMSE on validation set: {rmse:.4f}")

    submission_df = pd.read_csv("test.csv")
    submission_df = submission_df.dropna(subset=["Time_of_Entry"])
    submission_df = _engineer_features(submission_df)
    X_submission = submission_df.drop(columns=["Time_of_Entry"])
    y_submission = predict(model_bundle, X_submission)
    submit_predictions_for_test_set(y_submission)

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def compute_metrics_for_regression(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def submit_predictions_for_test_set(y_pred):
    answer_df = pd.read_csv("test.csv")
    y_true = answer_df["Time_of_Entry"].to_numpy()
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Prediction length mismatch: got {len(y_pred)} predictions for {len(y_true)} test rows."
        )
    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions must be finite numeric values.")

    rmse = compute_metrics_for_regression(y_true, y_pred)
    print(f"Final RMSE on test set: {rmse}.")

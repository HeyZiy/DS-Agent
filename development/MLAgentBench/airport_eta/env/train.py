import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = root_mean_squared_error(y_test, y_test_pred)
    return rmse


def train_model(X_train, y_train, X_valid, y_valid):
    # TODO. define and train the model
    # should return the trained model
    model = None
    return model


def predict(model, X):
    # TODO. predict the model
    # should return an array of predictions
    y_pred = np.random.randint(1, 11, len(X))
    return y_pred


if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['Time_of_Entry'])

    # Process data and store into numpy arrays.
    X = list(data_df.drop(['Time_of_Entry'], axis=1).to_numpy())
    y = data_df.Time_of_Entry.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    # should fill out the train_model function
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    # should fill out the predict function
    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print("final RMSE on validation set: ", rmse)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.dropna(subset=['Time_of_Entry'])
    X_submission = list(submission_df.drop(['Time_of_Entry'], axis=1).to_numpy())
    y_submission = predict(model, X_submission)
    submit_predictions_for_test_set(y_submission)

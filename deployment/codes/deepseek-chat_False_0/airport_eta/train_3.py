
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = root_mean_squared_error(y_test, y_test_pred) 
    return rmse

def preprocess_data(df, is_train=True, label_encoders=None, scaler=None):
    df = df.copy()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            else:
                df[col] = 0
    
    # Scale features
    if is_train:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
    else:
        scaled_features = scaler.transform(df)
    
    return scaled_features, label_encoders, scaler

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(X_train, y_train, X_valid, y_valid):
    # Preprocess data
    X_train_processed, label_encoders, scaler = preprocess_data(
        pd.DataFrame(X_train, columns=[f'col_{i}' for i in range(X_train.shape[1])]), 
        is_train=True
    )
    
    X_valid_processed, _, _ = preprocess_data(
        pd.DataFrame(X_valid, columns=[f'col_{i}' for i in range(X_valid.shape[1])]), 
        is_train=False, 
        label_encoders=label_encoders, 
        scaler=scaler
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid_processed).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    input_size = X_train_processed.shape[1]
    model = NeuralNetwork(input_size).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 100
    best_val_rmse = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_valid_tensor).cpu().numpy().flatten()
            val_rmse = compute_metrics_for_regression(y_valid, val_predictions)
        
        scheduler.step(val_rmse)
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val RMSE: {val_rmse:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Train ensemble models
    print("Training ensemble models...")
    
    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )
    xgb_model.fit(X_train_processed, y_train)
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )
    lgb_model.fit(X_train_processed, y_train)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=SEED
    )
    gb_model.fit(X_train_processed, y_train)
    
    # Store all models and preprocessing objects
    trained_model = {
        'nn_model': model,
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'gb_model': gb_model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'input_size': input_size
    }
    
    return trained_model

def predict(model_dict, X):
    # Preprocess input
    X_df = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
    X_processed, _, _ = preprocess_data(
        X_df, 
        is_train=False, 
        label_encoders=model_dict['label_encoders'], 
        scaler=model_dict['scaler']
    )
    
    # Neural network predictions
    nn_model = model_dict['nn_model']
    nn_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(device)
        nn_pred = nn_model(X_tensor).cpu().numpy().flatten()
    
    # Ensemble model predictions
    xgb_pred = model_dict['xgb_model'].predict(X_processed)
    lgb_pred = model_dict['lgb_model'].predict(X_processed)
    gb_pred = model_dict['gb_model'].predict(X_processed)
    
    # Weighted ensemble (adjust weights based on validation performance)
    weights = [0.3, 0.25, 0.25, 0.2]  # NN, XGB, LGB, GB
    y_pred = (weights[0] * nn_pred + 
              weights[1] * xgb_pred + 
              weights[2] * lgb_pred + 
              weights[3] * gb_pred)
    
    return y_pred

if __name__ == '__main__':
    # Load data
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['Time_of_Entry'])
    
    # Separate features and target
    X = data_df.drop(['Time_of_Entry'], axis=1)
    y = data_df['Time_of_Entry'].to_numpy()
    
    # Create a train-valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED
    )
    
    # Train model
    model = train_model(X_train, y_train, X_valid, y_valid)
    
    # Evaluate on validation set
    y_valid_pred = predict(model, X_valid.to_numpy())
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"Final RMSE on validation set: {rmse:.4f}")
    
    # Submit predictions for test set
    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.dropna(subset=['Time_of_Entry'])
    X_submission = submission_df.drop(['Time_of_Entry'], axis=1).to_numpy()
    y_submission = predict(model, X_submission)
    submit_predictions_for_test_set(y_submission)

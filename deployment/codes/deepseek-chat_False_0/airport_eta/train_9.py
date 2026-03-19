
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from submission import submit_predictions_for_test_set

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics_for_regression(y_test, y_test_pred):
    rmse = root_mean_squared_error(y_test, y_test_pred) 
    return rmse

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
    # Convert to numpy arrays if they aren't already
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # Try multiple models and select the best one
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse'
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            l2_leaf_reg=3,
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=50
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            random_state=SEED
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_rmse = float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        try:
            if name in ['XGBoost', 'CatBoost']:
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_valid_scaled, y_valid)],
                    verbose=False
                )
            else:
                model.fit(X_train_scaled, y_train)
            
            y_valid_pred = model.predict(X_valid_scaled)
            rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
            
            print(f"{name} RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    print(f"\nBest model: {best_model_name} with RMSE: {best_rmse:.4f}")
    
    # Also train a neural network for potential ensemble
    if len(X_train_scaled) > 1000:  # Only use NN if we have enough data
        try:
            # Prepare data for PyTorch
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
            X_valid_tensor = torch.FloatTensor(X_valid_scaled).to(device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # Initialize model
            input_size = X_train_scaled.shape[1]
            nn_model = NeuralNetwork(input_size).to(device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(nn_model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
            
            # Training loop
            best_nn_rmse = float('inf')
            patience = 20
            patience_counter = 0
            
            for epoch in range(200):
                nn_model.train()
                epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = nn_model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                nn_model.eval()
                with torch.no_grad():
                    y_valid_pred_nn = nn_model(X_valid_tensor).cpu().numpy()
                    nn_rmse = compute_metrics_for_regression(y_valid, y_valid_pred_nn)
                
                scheduler.step(nn_rmse)
                
                if nn_rmse < best_nn_rmse:
                    best_nn_rmse = nn_rmse
                    patience_counter = 0
                    best_nn_state = nn_model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Neural Network RMSE: {best_nn_rmse:.4f}")
            
            if best_nn_rmse < best_rmse:
                print("Neural Network is the best model!")
                # Create a wrapper for the neural network
                class NNWrapper:
                    def __init__(self, model, scaler):
                        self.model = model
                        self.scaler = scaler
                    
                    def predict(self, X):
                        X_scaled = self.scaler.transform(X)
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        self.model.eval()
                        with torch.no_grad():
                            predictions = self.model(X_tensor).cpu().numpy()
                        return predictions.flatten()
                
                nn_model.load_state_dict(best_nn_state)
                best_model = NNWrapper(nn_model, scaler)
                best_model_name = "NeuralNetwork"
        
        except Exception as e:
            print(f"Error training Neural Network: {e}")
    
    # Store scaler with the model for later use
    class ModelWithScaler:
        def __init__(self, model, scaler, model_name):
            self.model = model
            self.scaler = scaler
            self.model_name = model_name
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            if self.model_name == "NeuralNetwork":
                return self.model.predict(X_scaled)
            else:
                return self.model.predict(X_scaled)
    
    return ModelWithScaler(best_model, scaler, best_model_name)

def predict(model, X):
    X_array = np.array(X)
    y_pred = model.predict(X_array)
    return y_pred

if __name__ == '__main__':
    # Load and prepare data
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['Time_of_Entry'])
    
    # Handle missing values in features
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    data_df[numeric_cols] = data_df[numeric_cols].fillna(data_df[numeric_cols].median())
    
    # Process data and store into numpy arrays
    X = data_df.drop(['Time_of_Entry'], axis=1).to_numpy()
    y = data_df.Time_of_Entry.to_numpy()
    
    # Create a train-valid split of the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED
    )
    
    # Define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)
    
    # Evaluate the model on the valid set
    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"\nFinal RMSE on validation set: {rmse:.4f}")
    
    # Submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    
    # Handle missing values in test set
    submission_numeric_cols = submission_df.select_dtypes(include=[np.number]).columns
    submission_df[submission_numeric_cols] = submission_df[submission_numeric_cols].fillna(
        submission_df[submission_numeric_cols].median()
    )
    
    # Prepare test features
    X_submission = submission_df.drop(['Time_of_Entry'], axis=1).to_numpy()
    y_submission = predict(model, X_submission)
    
    submit_predictions_for_test_set(y_submission)

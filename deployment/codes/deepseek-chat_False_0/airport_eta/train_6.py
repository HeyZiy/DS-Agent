
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
from sklearn.impute import SimpleImputer
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


class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super(RegressionModel, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


def preprocess_data(X_train, X_valid, X_test=None):
    # Convert to DataFrame for easier processing
    train_df = pd.DataFrame(X_train)
    valid_df = pd.DataFrame(X_valid)
    
    # Identify categorical and numerical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    
    # Handle categorical features
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on train and transform all
            le.fit(pd.concat([train_df[col], valid_df[col]], axis=0).astype(str))
            train_df[col] = le.transform(train_df[col].astype(str))
            valid_df[col] = le.transform(valid_df[col].astype(str))
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    train_df[numerical_cols] = imputer.fit_transform(train_df[numerical_cols])
    valid_df[numerical_cols] = imputer.transform(valid_df[numerical_cols])
    
    # Scale numerical features
    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    valid_df[numerical_cols] = scaler.transform(valid_df[numerical_cols])
    
    # Convert back to numpy arrays
    X_train_processed = train_df.to_numpy().astype(np.float32)
    X_valid_processed = valid_df.to_numpy().astype(np.float32)
    
    return X_train_processed, X_valid_processed, scaler, imputer


def train_model(X_train, y_train, X_valid, y_valid, epochs=100, batch_size=64, lr=0.001):
    # Preprocess data
    X_train_processed, X_valid_processed, scaler, imputer = preprocess_data(X_train, X_valid)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_valid_tensor = torch.tensor(X_valid_processed, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train_processed.shape[1]
    model = RegressionModel(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    # Training loop
    for epoch in range(epochs):
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
            val_predictions = model(X_valid_tensor)
            val_loss = criterion(val_predictions, y_valid_tensor)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Store preprocessing objects for later use
    model.scaler = scaler
    model.imputer = imputer
    
    return model


def predict(model, X):
    model.eval()
    
    # Convert to DataFrame for preprocessing
    X_df = pd.DataFrame(X)
    
    # Handle categorical features if they exist
    categorical_cols = X_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            # Use label encoding (assuming similar distribution as training)
            le = LabelEncoder()
            # Fit on the current data (simplified approach)
            le.fit(X_df[col].astype(str))
            X_df[col] = le.transform(X_df[col].astype(str))
    
    # Handle missing values
    numerical_cols = X_df.select_dtypes(include=[np.number]).columns
    X_df[numerical_cols] = model.imputer.transform(X_df[numerical_cols])
    
    # Scale features
    X_df[numerical_cols] = model.scaler.transform(X_df[numerical_cols])
    
    # Convert to tensor
    X_processed = X_df.to_numpy().astype(np.float32)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor)
    
    return predictions.cpu().numpy()


if __name__ == '__main__':
    # Load data
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['Time_of_Entry'])
    
    # Process data and store into numpy arrays
    X = data_df.drop(['Time_of_Entry'], axis=1).to_numpy()
    y = data_df.Time_of_Entry.to_numpy()
    
    # Create a train-valid split of the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)
    
    # Define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid, epochs=150, batch_size=128, lr=0.001)
    
    # Evaluate the model on the valid set
    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"Final RMSE on validation set: {rmse:.4f}")
    
    # Submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.dropna(subset=['Time_of_Entry'])
    X_submission = submission_df.drop(['Time_of_Entry'], axis=1).to_numpy()
    y_submission = predict(model, X_submission)
    submit_predictions_for_test_set(y_submission)

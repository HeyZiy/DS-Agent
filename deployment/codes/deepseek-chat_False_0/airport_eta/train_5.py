
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
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
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
        return self.network(x)

def preprocess_data(X_train, X_valid, X_test=None):
    """Preprocess data with proper handling of categorical and numerical features"""
    # Convert to DataFrame for easier processing
    train_df = pd.DataFrame(X_train)
    valid_df = pd.DataFrame(X_valid)
    
    # Identify categorical columns (assuming object dtype or low cardinality)
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Handle categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on train and transform all
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        valid_df[col] = le.transform(valid_df[col].astype(str))
        encoders[col] = le
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    train_imputed = imputer.fit_transform(train_df)
    valid_imputed = imputer.transform(valid_df)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_imputed)
    X_valid_scaled = scaler.transform(valid_imputed)
    
    if X_test is not None:
        test_df = pd.DataFrame(X_test)
        for col in categorical_cols:
            if col in test_df.columns:
                le = encoders[col]
                test_df[col] = le.transform(test_df[col].astype(str))
        test_imputed = imputer.transform(test_df)
        X_test_scaled = scaler.transform(test_imputed)
        return X_train_scaled, X_valid_scaled, X_test_scaled, scaler, imputer, encoders
    
    return X_train_scaled, X_valid_scaled, scaler, imputer, encoders

def train_model(X_train, y_train, X_valid, y_valid):
    # Preprocess data
    X_train_processed, X_valid_processed, scaler, imputer, encoders = preprocess_data(
        X_train, X_valid
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid_processed).to(device)
    y_valid_tensor = torch.FloatTensor(y_valid).reshape(-1, 1).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    input_dim = X_train_processed.shape[1]
    model = RegressionModel(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(200):
        # Training phase
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
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Store preprocessing objects for later use
    model.scaler = scaler
    model.imputer = imputer
    model.encoders = encoders
    model.categorical_cols = list(encoders.keys())
    
    return model

def predict(model, X):
    model.eval()
    
    # Convert to DataFrame for preprocessing
    X_df = pd.DataFrame(X)
    
    # Handle categorical features
    for col in model.categorical_cols:
        if col in X_df.columns:
            le = model.encoders[col]
            # Handle unseen categories
            X_df[col] = X_df[col].astype(str)
            unseen_mask = ~X_df[col].isin(le.classes_)
            if unseen_mask.any():
                X_df.loc[unseen_mask, col] = le.classes_[0]
            X_df[col] = le.transform(X_df[col])
    
    # Impute and scale
    X_imputed = model.imputer.transform(X_df)
    X_scaled = model.scaler.transform(X_imputed)
    
    # Convert to tensor and predict
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()
    
    return predictions

if __name__ == '__main__':
    # Load and prepare data
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['Time_of_Entry'])
    
    # Separate features and target
    X = data_df.drop(['Time_of_Entry'], axis=1).values
    y = data_df['Time_of_Entry'].values
    
    # Create train-valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED
    )
    
    # Train model
    model = train_model(X_train, y_train, X_valid, y_valid)
    
    # Evaluate on validation set
    y_valid_pred = predict(model, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"Final RMSE on validation set: {rmse:.4f}")
    
    # Generate predictions for test set
    submission_df = pd.read_csv('test.csv')
    # Keep track of indices for rows with Time_of_Entry
    test_indices = submission_df.index[~submission_df['Time_of_Entry'].isna()].tolist()
    
    # Prepare test features
    X_submission = submission_df.drop(['Time_of_Entry'], axis=1).values
    y_submission = predict(model, X_submission)
    
    # Submit predictions
    submit_predictions_for_test_set(y_submission)


import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(X_train, y_train, X_valid, y_valid):
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid_processed).to(device)
    y_valid_tensor = torch.FloatTensor(y_valid).reshape(-1, 1).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    input_dim = X_train_processed.shape[1]
    model = RegressionModel(input_dim).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    n_epochs = 200
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(n_epochs):
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
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Return model and preprocessor for later use
    return {'model': model, 'preprocessor': preprocessor}

def predict(model_dict, X):
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    
    # Preprocess the data
    X_processed = preprocessor.transform(X)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_processed).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
    
    # Convert to numpy array and flatten
    return predictions.cpu().numpy().flatten()

if __name__ == '__main__':
    # Load data
    data_df = pd.read_csv('train.csv')
    data_df = data_df.dropna(subset=['Time_of_Entry'])
    
    # Separate features and target
    X = data_df.drop(['Time_of_Entry'], axis=1)
    y = data_df['Time_of_Entry'].values
    
    # Create a train-valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED
    )
    
    # Train the model
    model_dict = train_model(X_train, y_train, X_valid, y_valid)
    
    # Evaluate on validation set
    y_valid_pred = predict(model_dict, X_valid)
    rmse = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(f"Final RMSE on validation set: {rmse:.4f}")
    
    # Submit predictions for test set
    submission_df = pd.read_csv('test.csv')
    submission_df = submission_df.dropna(subset=['Time_of_Entry'])
    X_submission = submission_df.drop(['Time_of_Entry'], axis=1)
    y_submission = predict(model_dict, X_submission)
    submit_predictions_for_test_set(y_submission)

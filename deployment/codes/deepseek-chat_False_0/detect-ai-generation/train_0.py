
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from submission import submit_predictions_for_test_set
from tqdm import tqdm

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics_for_classification(y_test, y_test_pred):
    acc = accuracy_score(y_test, y_test_pred) 
    return acc

def train_model(X_train, y_train, X_valid, y_valid):
    # Load pre-trained model and tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.to(device)
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    valid_dataset = TextDataset(X_valid, y_valid, tokenizer)
    
    # Create dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Training parameters
    epochs = 3
    learning_rate = 2e-5
    num_training_steps = len(train_loader) * epochs
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        
        valid_acc = accuracy_score(valid_labels, valid_preds)
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Valid Acc = {valid_acc:.4f}')
    
    return model, tokenizer

def predict(model, tokenizer, X, batch_size=32):
    model.eval()
    predictions = []
    
    # Create dataset and dataloader
    dataset = TextDataset(X, [0]*len(X), tokenizer)  # dummy labels
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)

if __name__ == '__main__':
    data_df = pd.read_csv('train.csv')
    
    # Process data and store into numpy arrays.
    X = data_df.text.to_numpy()
    y = data_df.label.to_numpy()

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # define and train the model
    model, tokenizer = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set
    y_valid_pred = predict(model, tokenizer, X_valid)
    acc = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("Final Accuracy on validation set: ", acc)

    # submit predictions for the test set
    submission_df = pd.read_csv('test.csv')
    X_submission = submission_df.text.to_numpy()
    y_submission = predict(model, tokenizer, X_submission)
    submit_predictions_for_test_set(y_submission)

from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from datetime import datetime
import argparse
from torch.utils.data import DataLoader

def load_data():
    sql = """
    SELECT *
    FROM RicardoInterview.product_detection_training_data
    """

    df_load = load_bq_data(sql)
    print(df_load.head())
    return df_load

def load_bq_data(_sql):
    client_bq = bigquery.Client.from_service_account_json("../credentials.json", project='charged-dialect-824')

    _df = client_bq.query(_sql).to_dataframe()
    return _df

def preprocess_data(df):
    # Clean text
    df['title'] = df['title'].str.lower().str.replace('[^\w\s]', '', regex=True).fillna('')
    df['subtitle'] = df['subtitle'].str.lower().str.replace('[^\w\s]', '', regex=True).fillna('')
    df['combined_text'] = df['title'] + ' ' + df['subtitle']

    # Encode labels
    le = LabelEncoder()
    df['productType_encoded'] = le.fit_transform(df['productType'])
    return df

def split_data(df):
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['combined_text'],
        df['productType_encoded'],
        test_size=0.2,
        stratify=df['productType_encoded'],
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1,  # 10% of the remaining 80% (i.e., 8% of total)
        stratify=y_temp,
        random_state=42
    )

    # Save split
    X_train.to_csv('X_train.csv', index=False)
    X_val.to_csv('X_val.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_val.to_csv('y_val.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    print(type(X_train))
    print(type(y_train))
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'f1': f1}

# Create dataset class
class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
        
def main():
    df_load = load_data()

    df_load = preprocess_data(df_load)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_load)

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df_load['productType'].unique()))

    # Tokenize data
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

    train_dataset = ProductDataset(train_encodings, y_train.tolist())
    val_dataset = ProductDataset(val_encodings, y_val.tolist())
    test_dataset = ProductDataset(test_encodings, y_test.tolist())
    
    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=300,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=100,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model
    model.save_pretrained(f'distilbert-base-uncased-finetuned-product-detection-{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    tokenizer.save_pretrained(f'distilbert-base-uncased-finetuned-product-detection-{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # Evaluate model
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    # Save test results
    with open('test_results.txt', 'w') as f:
        f.write(str(test_results))

def evaluate():
    # load checkpoint
    model = DistilBertForSequenceClassification.from_pretrained('results/checkpoint-76500')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # load test data
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    print(y_test.head())

    # tokenize test data
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)
    test_dataset = ProductDataset(test_encodings, y_test.tolist())

    # DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=64)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # calculate f1 score
    f1 = f1_score(y_test, all_preds, average='weighted')
    print(f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        main()
    else:
        evaluate()

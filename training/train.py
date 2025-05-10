from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
import joblib
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn

def load_data():
    sql = """
    SELECT *
    FROM RicardoInterview.product_detection_training_data
    """

    df_load = load_bq_data(sql)
    print(df_load.head())
    df_load.to_csv('data.csv', index=False)
    return df_load

def load_bq_data(_sql):
    client_bq = bigquery.Client.from_service_account_json("../credentials.json", project='charged-dialect-824')

    _df = client_bq.query(_sql).to_dataframe()
    return _df

def preprocess_data(df):
    # Before cleaning
    print(df.shape)

    # Clean text
    df['title'] = df['title'].str.lower().str.replace('[^\w\s]', '', regex=True).fillna('')
    df['subtitle'] = df['subtitle'].str.lower().str.replace('[^\w\s]', '', regex=True).fillna('')

    # Remove rows with empty title or subtitle
    df = df[df['title'].notna() & df['subtitle'].notna()]
    df = df[df['title'] != '']
    df = df[df['productType'] != '']

    # Remove rows with empty productType
    df = df[df['productType'].notna()]

    # remove space in beginning and end of title and subtitle
    df['title'] = df['title'].str.strip()
    df['subtitle'] = df['subtitle'].str.strip()

    # remove space in beginning and end of productType
    df['productType'] = df['productType'].str.strip()
    
    df['combined_text'] = df['title'] + ' ' + df['subtitle']

    # After cleaning
    print(df.shape)

    # Encode labels
    le = LabelEncoder()
    df['productType_encoded'] = le.fit_transform(df['productType'])
    joblib.dump(le, "label_encoder.joblib")
    with open("class_names.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
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

    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='weighted')
    rec = recall_score(labels, predictions, average='weighted')
    return {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec}

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.padding_side = 'right'
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df_load['productType'].unique()))

    # Tokenize data
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=140)
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=140)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=140)

    train_dataset = ProductDataset(train_encodings, y_train.tolist())
    val_dataset = ProductDataset(val_encodings, y_val.tolist())
    test_dataset = ProductDataset(test_encodings, y_test.tolist())
    
    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=100,
        save_total_limit=2,
        eval_steps=500,
        eval_strategy="steps"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save model
    model.save_pretrained(f'bert-base-uncased-finetuned-product-detection-{datetime_str}')
    tokenizer.save_pretrained(f'bert-base-uncased-finetuned-product-detection-{datetime_str}')

    # Evaluate model
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    # Save test results
    with open('test_results.txt', 'w') as f:
        f.write(str(test_results))

def evaluate():
    df_load = load_data()

    # load checkpoint
    model = BertForSequenceClassification.from_pretrained("/home/vankhoa@median.cad/code/github/di-interview-product-classifier/training/bert-base-uncased-finetuned-product-detection-20250509_022000")

    tokenizer = BertTokenizer.from_pretrained('/home/vankhoa@median.cad/code/github/di-interview-product-classifier/training/bert-base-uncased-finetuned-product-detection-20250509_022000')
    tokenizer.padding_side = 'right'

    print(pd.read_csv('X_test.csv').shape)
    print(pd.read_csv('y_test.csv').shape)

    # load test data
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    y_test = y_test['productType_encoded']
    X_test = X_test['combined_text']

    print(y_test.head())
    print(X_test.head())

    assert len(X_test) == len(y_test), f"Lengths do not match: {len(X_test)} != {len(y_test)}"

    # tokenize test data
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=140)

    test_dataset = ProductDataset(test_encodings, y_test.tolist())

    # DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=16)

    model.eval()
    all_preds = []
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Top-1 predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

            # Top-3 predictions
            top3 = torch.topk(logits, k=3, dim=1).indices.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(labels_np.shape[0]):
                if labels_np[i] in top3[i]:
                    top3_correct += 1
                total += 1

    # Standard metrics for top-1
    f1 = f1_score(y_test, all_preds, average='weighted')
    acc = accuracy_score(y_test, all_preds)
    prec = precision_score(y_test, all_preds, average='weighted')
    rec = recall_score(y_test, all_preds, average='weighted')

    # Top-3 accuracy
    top3_acc = top3_correct / total
    print(f"F1 score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"Top-3 Accuracy: {top3_acc}")

    # Save results
    with open('results.txt', 'w') as f:
        f.write(f"F1 score: {f1}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"Top-3 Accuracy: {top3_acc}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        main()
    else:
        evaluate()

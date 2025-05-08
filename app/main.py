from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import re
import json

# Set your model directory here
MODEL_DIR = "../training/results/checkpoint-76500"  # <-- CHANGE THIS TO YOUR ACTUAL MODEL DIR
CLASS_NAMES_PATH = "../training/class_names.json"  # <-- set your class names path

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()

# Load class names
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

app = FastAPI()

class ClassifyRequest(BaseModel):
    title: str
    subtitle: str = ""

# Preprocessing function (same as in training)
def preprocess_text(title, subtitle):
    title = title.lower()
    subtitle = subtitle.lower()
    title = re.sub(r"[^\w\s]", "", title)
    subtitle = re.sub(r"[^\w\s]", "", subtitle)
    return title + " " + subtitle

@app.post("/classify")
def classify(req: ClassifyRequest):
    title = preprocess_text(req.title, req.subtitle)
    # Tokenize
    inputs = tokenizer(title, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

        # Prepare top 3 results
        top_3_results = [
            {
                "product_type": class_names[top3_indices[0, i].item()],
                "score": float(top3_probs[0, i].item())
            }
            for i in range(3)
        ]
        # Top-1
        product_type = class_names[top3_indices[0, 0].item()]

    return {
        "title": req.title,
        "top_3_results": top_3_results,
        "product_type": product_type
    }

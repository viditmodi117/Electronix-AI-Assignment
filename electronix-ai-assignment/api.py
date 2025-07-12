from fastapi import FastAPI, Request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import asyncio

app = FastAPI()

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("./model").to(device)
tokenizer = BertTokenizer.from_pretrained("./model")
model.eval()

class PredictionRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Async batch processing
    inputs = tokenizer(request.texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Map predictions to labels
    labels = ["negative" if pred == 0 else "positive" for pred in predictions]
    return {"predictions": labels}

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API"}

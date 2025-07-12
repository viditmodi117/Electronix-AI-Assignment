import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import time
import os

def fine_tune_model():
    # Check device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset("imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Subset for demo
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        disable_tqdm=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Measure fine-tuning time
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Save model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

    print(f"Fine-tuning time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    fine_tune_model()

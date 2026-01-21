import os
import pandas as pd
import joblib
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "bert_email_classifier")
CODE_DIR = os.path.join(PROJECT_ROOT, "Code")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

print(f"[INFO] Project root: {PROJECT_ROOT}")
print(f"[INFO] Data dir: {DATA_DIR}")
print(f"[INFO] Loading training data from: {TRAIN_CSV}")


# Load Data
df = pd.read_csv(TRAIN_CSV, encoding="latin1")

df["subject"] = df["subject"].fillna("")
df["email_body"] = df["email_body"].fillna("")
df["text"] = df["subject"] + " " + df["email_body"]

df["label"] = df["label"].map({"job": 1, "non_job": 0})

# Balance Dataset
job_df = df[df["label"] == 1]
nonjob_df = df[df["label"] == 0]

min_count = min(len(job_df), len(nonjob_df))
job_df = job_df.sample(min_count, random_state=42)
nonjob_df = nonjob_df.sample(min_count, random_state=42)

df_balanced = pd.concat([job_df, nonjob_df]).sample(frac=1, random_state=42)
print(f"[INFO] Balanced dataset size: {len(df_balanced)}")
dataset = Dataset.from_pandas(df_balanced[["text", "label"]])

# Model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize, batched=True)

#Train
args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()

# Model Saved in Code file
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"[INFO] Saved HuggingFace model to: {MODEL_DIR}")


class EmailClassifierWrapper:
    def __init__(self, model_path=MODEL_DIR):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()
        label = int(torch.argmax(logits))
        return label, prob

# Save pickle
wrapper = EmailClassifierWrapper()
PICKLE_PATH = os.path.join(CODE_DIR, "bert_email_classifier.pkl")
joblib.dump(wrapper, PICKLE_PATH)

print(f"[INFO] Pickle wrapper saved to: {PICKLE_PATH}")

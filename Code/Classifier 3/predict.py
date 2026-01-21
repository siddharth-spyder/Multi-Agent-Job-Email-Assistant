import os
import pandas as pd
import joblib
import torch


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
CODE_DIR = os.path.join(PROJECT_ROOT, "Code")

MODEL_PKL = os.path.join(CODE_DIR, "bert_email_classifier.pkl")
INPUT_FILE = os.path.join(DATA_DIR, "gmail_subject_body_date.xlsx")
OUTPUT_FILE = os.path.join(DATA_DIR, "mail_classified3.xlsx")

print(f"[INFO] Loading model from: {MODEL_PKL}")

class EmailClassifierWrapper:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

wrapper = joblib.load(MODEL_PKL)
model = wrapper.model
tokenizer = wrapper.tokenizer
model.eval()

print("[INFO] Model loaded.")

df = pd.read_excel(INPUT_FILE)

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

print(f"[INFO] Loaded {len(df)} samples for prediction.")

preds = []
probs = []

for text in df["text"]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()
        pred = int(torch.argmax(logits))
    preds.append(pred)
    probs.append(prob)

df["job_label"] = preds
df["prob_job"] = probs
df["predicted_label"] = df["job_label"].map({1: "job", 0: "non_job"})

print("[INFO] Predictions complete.")

df.to_excel(OUTPUT_FILE, index=False)
print(f"[INFO] Saved predictions to: {OUTPUT_FILE}")

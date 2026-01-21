import pandas as pd
import joblib
import os
import sys

# -----------------------------
# CONFIG
# -----------------------------
excel_path = "../../data/gmail_subject_body_date.xlsx"
model_path = "job_classifier_baseline.pkl"


# -----------------------------
# SAFE LOADER FOR EXCEL FILES
# -----------------------------
def load_excel_safely(path):
    try:
        print(f"[INFO] Loading Excel file: {path}")
        return pd.read_excel(path)
    except Exception as e:
        print(f"[ERROR] Could not read Excel file: {e}")
        sys.exit(1)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
if not os.path.exists(excel_path):
    print(f"[ERROR] File not found: {excel_path}")
    sys.exit(1)

df = load_excel_safely(excel_path)

# Validate required columns
required_cols = {"subject", "body"}
if not required_cols.issubset(df.columns):
    print(f"[ERROR] Excel must contain columns {required_cols}")
    print("[ERROR] Columns found:", set(df.columns))
    sys.exit(1)

# -----------------------------
# 2. PREPARE TEXT
# -----------------------------
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

print(f"[INFO] Loaded {len(df)} uncleaned emails.")

# -----------------------------
# 3. LOAD TRAINED MODEL
# -----------------------------
if not os.path.exists(model_path):
    print(f"[ERROR] Model file not found: {model_path}")
    sys.exit(1)

print(f"[INFO] Loading model: {model_path}")
model = joblib.load(model_path)

# -----------------------------
# 4. PREDICT ON UNCLEANED DATASET
# -----------------------------
print("[INFO] Running predictions...")
df["job_label"] = model.predict(df["text"])

# Optional: prediction probabilities
df["prob_job"] = model.predict_proba(df["text"])[:, 1]

print("[INFO] Prediction complete.")

# -----------------------------
# 5. SAVE RESULTS
# -----------------------------
output_file = "../../Data/mail_classified1.xlsx"
df.to_excel(output_file, index=False)

print(f"[INFO] Predictions saved to {output_file}")
print(df[["subject", "job_label", "prob_job"]].head(10))
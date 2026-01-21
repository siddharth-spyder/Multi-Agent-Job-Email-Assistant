import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os


def load_csv_safely(path):
    """
    Try multiple encodings until one works.
    """
    encodings_to_try = [
        "utf-8",
        "latin1",
        "cp1252",
        "utf-8-sig",
    ]

    for enc in encodings_to_try:
        try:
            print(f"[INFO] Attempting to read CSV using encoding: {enc}")
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            print(f"[WARN] Failed with encoding {enc}: {e}")


    print("[INFO] Falling back to utf-8 with errors='replace'")
    return pd.read_csv(path, encoding="utf-8", errors="replace")


# Load Data
csv_path = "../../data/train.csv"

if not os.path.exists(csv_path):
    print(f"[ERROR] File not found: {csv_path}")
    sys.exit(1)

df = load_csv_safely(csv_path)

required_cols = {"subject", "email_body", "label"}
if not required_cols.issubset(df.columns):
    print(f"[ERROR] CSV must contain columns: {required_cols}")
    print(f"[ERROR] Found columns: {set(df.columns)}")
    sys.exit(1)


print("\n===== LABEL DISTRIBUTION =====\n")
print(df["label"].value_counts())

print("\n===== LABEL DISTRIBUTION (PERCENT) =====\n")
print(df["label"].value_counts(normalize=True) * 100)


df["subject"] = df["subject"].fillna("")
df["email_body"] = df["email_body"].fillna("")
df["text"] = df["subject"] + " " + df["email_body"]

print(f"[INFO] Loaded {len(df)} rows.")



# Count classes
job_count = df[df["label"] == "job"].shape[0]
non_job_count = df[df["label"] == "non_job"].shape[0]

print(f"[INFO] Before balancing: job={job_count}, non_job={non_job_count}")

# Downsample non_job to match job count
non_job_df = df[df["label"] == "non_job"].sample(job_count, random_state=42)
job_df = df[df["label"] == "job"]

df = pd.concat([job_df, non_job_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"[INFO] After balancing: job={job_df.shape[0]}, non_job={non_job_df.shape[0]}")
print(f"[INFO] New dataset size: {len(df)}")


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Tf-IDF and Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=300,
        C=2.0,
        class_weight="balanced"
    ))
])

print("[INFO] Training model...")
pipeline.fit(X_train, y_train)
print("[INFO] Training complete.")

# Evaluation 

print("\n===== CLASSIFICATION REPORT =====\n")
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

print("\n===== CONFUSION MATRIX =====\n")
print(confusion_matrix(y_test, preds))


# Model Save

model_path = "job_classifier_baseline.pkl"
joblib.dump(pipeline, model_path)

print(f"\n[INFO] Model saved to {model_path}")
print("[INFO] Done.")

import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# CONFIG
# ============================

JOBS_FILE   = "../../Data/jobs.csv"
GMAIL_FILE  = "../../Data/gmail_subject_body_date.xlsx"
OUTPUT_FILE = "../../Data/mail_classified2.xlsx"

JOBS_SUBJECT_COL = "subject"
JOBS_BODY_COL = "email_body"
GMAIL_SUBJECT_COL = "subject"
GMAIL_BODY_COL = "body"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# STRICT threshold to avoid alerts being called "job"
VERY_HIGH_SIM_THRESHOLD = 0.82   # you can tune 0.80–0.85 if needed


# ============================
# PATTERNS
# ============================

# 1. CLEAR job-process emails (force job)
JOB_PROCESS_PATTERNS = [
    # application submitted / received / sent
    r"\bwe received your application\b",
    r"\bwe have received your application\b",
    r"\byour application has been received\b",
    r"\byour application was sent\b",
    r"\bapplication was sent to\b",
    r"\bapplication submitted\b",
    r"\bapplication confirmation\b",

    # generic thank you for applying / submission
    r"\bthank you for applying\b",
    r"\bthanks for applying\b",
    r"\bthank you for your application\b",
    r"\bthank you for your online submission\b",
    r"\bthank you\b.*\bsubmission\b",

    r"\bstatus of your application\b",
    r"\bapplication status\b",
    r"\bupdate on your application\b",
    r"\bregarding your application\b",
    r"\babout your application\b",
    r"\bwe are reviewing your application\b",
    r"\bwe are currently reviewing\b.*\bapplication\b",

    # "we just received your ..." style
    r"\bwe just received\b.*\b(your information|your application|your submission)\b",

    # explicit "your application" + position/role
    r"\byour application\b.*\b(position|role|opening|opportunity)\b",
    r"\bapplication\b.*\b(position|role|opening|opportunity)\b",

    # interview / assessment
    r"\binterview invitation\b",
    r"\bwe would like to invite you to interview\b",
    r"\binvite you to interview\b",
    r"\bschedule your interview\b",
    r"\binterview has been scheduled\b",
    r"\bphone interview\b",
    r"\bphone screen\b",
    r"\bvideo interview\b",
    r"\bonline assessment\b",
    r"\btechnical assessment\b",
    r"\bcoding test\b",
    r"\bcase study\b",
    r"\bhiring process\b",
    r"\brecruitment process\b",

    # rejections
    r"\bwe regret to inform you\b",
    r"\bafter careful consideration\b",
    r"\bwe will not be moving forward with your application\b",
    r"\bnot moving forward with your application\b",
    r"\bapplication was not successful\b",

    # offers
    r"\bjob offer\b",
    r"\boffer of employment\b",
    r"\bwe are pleased to offer you\b",
    r"\bpleased to offer you the position\b",
    r"\bwe are happy to offer\b",
    r"\bwe are excited to offer\b",
    r"\boffer letter\b",
]

# 2. CLEAR job-alerts (force non_job)
JOB_ALERT_PATTERNS = [
    r"\bjob alert\b",
    r"\bjob alerts\b",
    r"\bnew jobs for you\b",
    r"\brecommended jobs\b",
    r"\bjobs you may be interested\b",
    r"\bjobs you might be interested\b",
    r"\bview similar jobs\b",
    r"\bview more jobs\b",
    r"\bweekly job alerts\b",
    r"\bdaily job alerts\b",
    r"\bmatching jobs\b",
    r"\bmatched jobs\b",
    r"\bjob recommendations\b",
    r"\bjob suggestions\b",
    r"\bexplore more jobs\b",
    r"\btop job picks\b",
    r"\bpopular jobs\b",
    r"\bsearch more jobs\b",
    r"\bnew job opportunities for you\b",
]

# 3. Keywords that must exist to trust a similarity-based "job"
CORE_JOB_KEYWORDS = [
    "your application",
    "we received your application",
    "thank you for applying",
    "thanks for applying",
    "thank you for your application",
    "thank you for your online submission",
    "application status",
    "thank you for applying",
    "thank you for taking assessment",
    "status of your application",
    "update on your application",
    "interview",
    "assessment",
    "phone screen",
    "video interview",
    "offer",
    "job offer",
    "not moving forward",
    "regret to inform",
    "we just received your information",
    "we just received your application",
    "we just received your submission",
]


def normalize(t: str) -> str:
    if t is None:
        return ""
    t = str(t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def safe_str(x) -> str:
    """Convert any value (including NaN) safely to string."""
    if pd.isna(x):
        return ""
    return str(x)


def combine(subject, body) -> str:
    s = safe_str(subject)
    b = safe_str(body)
    return normalize(s + "\n" + b)


def matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def contains_keyword(text: str, keywords) -> bool:
    return any(k in text for k in keywords)


def main():
    print("Loading datasets...")

    if not os.path.exists(JOBS_FILE):
        raise FileNotFoundError(f"jobs file not found: {JOBS_FILE}")
    if not os.path.exists(GMAIL_FILE):
        raise FileNotFoundError(f"Gmail file not found: {GMAIL_FILE}")

    # ---- FIX: robust CSV reading for jobs.csv ----
    try:
        df_jobs = pd.read_csv(JOBS_FILE)
    except UnicodeDecodeError:
        # fallback for mixed encodings / special chars
        df_jobs = pd.read_csv(JOBS_FILE, encoding="latin1")

    df_gmail = pd.read_excel(GMAIL_FILE)

    # Sanity checks
    if JOBS_SUBJECT_COL not in df_jobs.columns or JOBS_BODY_COL not in df_jobs.columns:
        raise ValueError(
            f"jobs.csv must contain columns '{JOBS_SUBJECT_COL}' and '{JOBS_BODY_COL}'"
        )
    if GMAIL_SUBJECT_COL not in df_gmail.columns or GMAIL_BODY_COL not in df_gmail.columns:
        raise ValueError(
            f"Gmail file must contain columns '{GMAIL_SUBJECT_COL}' and '{GMAIL_BODY_COL}'"
        )

    # Combine subject+body → full training text
    df_jobs["full_text"] = df_jobs.apply(
        lambda r: combine(r[JOBS_SUBJECT_COL], r[JOBS_BODY_COL]),
        axis=1,
    )
    df_jobs = df_jobs[df_jobs["full_text"].str.strip() != ""]

    # Combine subject+body → full Gmail text
    df_gmail["full_text"] = df_gmail.apply(
        lambda r: combine(r[GMAIL_SUBJECT_COL], r[GMAIL_BODY_COL]),
        axis=1,
    )

    print(f"Loaded {len(df_jobs)} training job emails.")
    print(f"Loaded {len(df_gmail)} Gmail emails.\n")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding job examples...")
    job_emb = model.encode(
        df_jobs["full_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Encoding Gmail messages...")
    gmail_emb = model.encode(
        df_gmail["full_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Calculating similarities...")
    sim_matrix = cosine_similarity(gmail_emb, job_emb)
    max_sims = sim_matrix.max(axis=1)

    final_labels = []

    for text, sim in zip(df_gmail["full_text"], max_sims):
        # 1) Strong job-process → always job
        if matches_any(JOB_PROCESS_PATTERNS, text):
            final_labels.append("job")
            continue

        # 2) Strong job-alert → always non_job
        if matches_any(JOB_ALERT_PATTERNS, text):
            final_labels.append("non_job")
            continue

        # 3) Strict similarity logic (rescue true jobs only)
        if sim >= VERY_HIGH_SIM_THRESHOLD and contains_keyword(text, CORE_JOB_KEYWORDS):
            final_labels.append("job")
        else:
            final_labels.append("non_job")

    df_gmail["job_label"] = final_labels

    df_gmail.to_excel(OUTPUT_FILE, index=False)
    print(f"\n[✓] Saved to {OUTPUT_FILE}")
    print("Classification complete — ONE column added: job_label\n")


if __name__ == "__main__":
    main()

import os
import json
import re
import requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

SYSTEM_PROMPT = """
You are an assistant that extracts structured job application information from email text.

You MUST return a single JSON object with exactly these keys:
  - "company_name": string
  - "position_applied": string
  - "application_date": string
  - "status": string

GENERAL RULES
- Use ONLY text that appears in the email subject or body (or their provided summary).
- Do NOT invent company names or job titles.
- For company_name and position_applied, ALWAYS copy contiguous spans from the given text.
- Do NOT rephrase or add extra words like "role", "position", or "at Company" unless they literally appear.
- If information is missing or unclear, set the field to "" (empty string), except status.

FIELD DEFINITIONS

1) "company_name":
   - The employer company name, e.g. "IBM", "Google", "Adobe".
   - If you see variants like "Careers at Adobe" or "IBM Careers":
       - Prefer the clean company name if it appears on its own.
       - Otherwise, use the shortest phrase that clearly includes the company name.
   - If multiple companies are mentioned and it is unclear which is the employer, use "".

2) "position_applied":
   - The job title or position the candidate applied for, copied exactly from the given text if present.
   - Look especially in:
       - the subject line,
       - any "position:", "role:", or "job title:" phrases,
       - sentences around "applied for", "application for", or "your application for".
   - If you are not reasonably sure which phrase is the job title, use "".

3) "application_date":
   - A string in format "YYYY-MM-DD", or "" if unknown.
   - If the application date is not explicitly mentioned, you MAY leave this as "".
   - Do NOT invent dates.

4) "status":
   - Must be exactly one of:
       "applied", "in progress", "rejected", "job offered"
   - Map based on the strongest clues in the text:
       • "job offered": clear offer made.
       • "rejected": clearly rejected / not moving forward.
       • "in progress": under review, assessments, interviews, still in consideration.
       • "applied": confirmation / submission received, no next steps yet.
   - You MUST always choose exactly one of these four values.

OUTPUT:
- Return ONLY a single JSON object like:
  {
    "company_name": "...",
    "position_applied": "...",
    "application_date": "...",
    "status": "applied"
  }
- Do NOT include explanations, markdown, or code fences.
"""

SUMMARY_PROMPT = """
You are an assistant that rewrites job-application related emails into a concise,
human-readable summary WITHOUT losing any important job-related information.

INPUT: email subject and body.
OUTPUT: one paragraph of text called "mailcontent".

Rules:
- Keep ALL important job-related details:
    - company names
    - job titles / positions
    - dates and times
    - application IDs or reference numbers
    - interview / assessment details
    - decisions (applied, under review, rejected, offered, next steps)
- You MAY remove:
    - long signatures
    - unsubscribe / footer / legal boilerplate
    - marketing fluff that does not change the meaning of the application status
- When you mention company name or job title, COPY the exact phrase from the email where possible.
- Do NOT invent or guess any information.
- Do NOT output JSON or bullet points. Just a single, clean paragraph of text.
"""

def matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def heuristic_position(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    text_all = subject + "\n" + body

    def _clean_title(t: str) -> str:
        t = t.strip(" .|,-–—:;")
        t = t.replace("*", "").replace('"', "").replace("'", "")
        t = re.sub(r"\b(position|role|job)\b.*$", "", t, flags=re.IGNORECASE)
        t = re.split(r"\bat\b", t, flags=re.IGNORECASE)[0]
        t = re.split(r"\bwith\b", t, flags=re.IGNORECASE)[0]
        t = re.sub(r"^(the)\s+", "", t, flags=re.IGNORECASE)
        t = " ".join(t.split())
        return t.strip()

    patterns = [
        r"application for\s+(?P<title>[^,\n]+)",
        r"applied for the position of\s+(?P<title>[^,\n]+)",
        r"applied for\s+(?P<title>[^,\n]+)",
        r"for the position of\s+(?P<title>[^,\n]+)",
        r"position:\s*(?P<title>[^,\n]+)",
        r"role:\s*(?P<title>[^,\n]+)",
        r"job title:\s*(?P<title>[^,\n]+)",
        r"position applied:\s*(?P<title>[^,\n]+)",
    ]

    for pat in patterns:
        m = re.search(pat, text_all, flags=re.IGNORECASE)
        if m:
            title = _clean_title(m.group("title"))
            if title and 1 <= len(title.split()) <= 20:
                return title

    ref_match = re.search(
        r"ref:\s*\S+\s*[-–]\s*(?P<title>[^,\n]+)",
        text_all,
        flags=re.IGNORECASE,
    )
    if ref_match:
        title = _clean_title(ref_match.group("title"))
        if title and 1 <= len(title.split()) <= 20:
            return title

    bad_words = [
        "thank you",
        "thanks",
        "thanks for applying",
        "thanks for your interest",
        "application",
        "applied",
        "confirmation",
        "update",
        "status",
        "candidate acknowledgement",
        "profile submitted",
        "careers",
        "newsletter",
        "feedback",
        "survey",
    ]
    subj_lower = subject.lower()
    if subject and len(subject.split()) <= 10 and not any(b in subj_lower for b in bad_words):
        title = _clean_title(subject)
        if title and len(title.split()) >= 1:
            return title

    return ""

def clean_final_position(title: str) -> str:
    if not title:
        return ""

    t = title.strip()
    t = re.sub(r"\bposition is\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bis currently under review\b.*$", "", t, flags=re.IGNORECASE)
    t = t.strip(" .|,-–—")

    if re.fullmatch(r"(the|this|that|a|an)?\s*(position|role|job)", t, flags=re.IGNORECASE):
        return ""

    t = re.sub(r"^(the)\s+", "", t, flags=re.IGNORECASE).strip()
    return t

def infer_status(full_text: str, llm_status: str) -> str:
    text = full_text.lower()

    offer_patterns = [
        r"\bjob offer\b",
        r"\boffer of employment\b",
        r"\bwe are pleased to offer you\b",
        r"\bwe are excited to offer\b",
        r"\boffer letter\b",
    ]
    if matches_any(offer_patterns, text):
        return "job offered"

    reject_patterns = [
        r"\bwe regret to inform you\b",
        r"\bnot be moving forward with your application\b",
        r"\bwe will not be moving forward\b",
        r"\bnot moving forward with your application\b",
        r"\bapplication was not successful\b",
        r"\bhas been unsuccessful\b",
        r"\bunfortunately[, ]+we\b.*\bnot\b.*\bmove forward\b",
        r"\bno further action will be taken on your online submission\b",
    ]
    if matches_any(reject_patterns, text):
        return "rejected"

    in_progress_patterns = [
        r"\byou are still in consideration\b",
        r"\bstill in consideration\b",
        r"\bstill considered\b",
        r"\bunder review\b",
        r"\bcurrently under review\b",
        r"\bwe are reviewing your application\b",
        r"\bwe will provide an update\b",
        r"\bwe will contact you\b.*\bnext steps\b",
        r"\bnext steps\b",
        r"\bassessment\b",
        r"\bcoding assessment\b",
        r"\bon-demand assessments\b",
        r"\bassessment centre\b",
        r"\bvideo interview\b",
        r"\bpre[- ]?recorded video interview\b",
        r"\bprvi\b",
        r"\binvited to complete\b.*(assessment|interview)",
        r"\binvited to take\b.*(assessment|interview)",
        r"\bselected to complete\b.*(assessment|interview)",
        r"\bselected to apply\b",
    ]
    if matches_any(in_progress_patterns, text):
        return "in progress"

    applied_patterns = [
        r"\bapplication has been submitted\b",
        r"\byour application has been submitted\b",
        r"\bhas been submitted successfully\b",
        r"\bwe received your application\b",
        r"\bwe have received your application\b",
        r"\bthank you for applying\b",
        r"\bthanks for applying\b",
        r"\bthank you for your application\b",
        r"\bapplication received\b",
        r"\byour submission will be reviewed\b",
        r"\bapplication is currently being reviewed\b",
    ]
    if matches_any(applied_patterns, text):
        return "applied"

    if llm_status in {"applied", "in progress", "rejected", "job offered"}:
        return llm_status
    return "applied"


def call_ollama(prompt: str, model: str = MODEL_NAME, url: str = OLLAMA_URL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Ollama call failed: {e}")
        return ""

def extract_json_object(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    return cleaned

def summarize_email(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    body_short = body[:8000]

    user_prompt = f"""
{SUMMARY_PROMPT}

Email subject:
\"\"\"{subject}\"\"\"

Email body:
\"\"\"{body_short}\"\"\"

Now return ONLY the summarized mailcontent as a single paragraph of plain text.
"""
    summary = call_ollama(user_prompt)
    if not summary:
        combined = (subject + " " + body).strip()
        return combined[:4000]
    return summary.strip()

def call_llm_extract(subject: str, mailcontent: str, date_received: str) -> dict:
    mailcontent = mailcontent or ""
    text_for_extraction = subject + "\n" + mailcontent

    user_prompt = f"""
{SYSTEM_PROMPT}

Here is the email content you must extract from:

\"\"\"{text_for_extraction}\"\"\"

Email received date (string, may be in various formats):
\"\"\"{date_received or ""}\"\"\"

Return ONLY a JSON object with keys:
company_name, position_applied, application_date, status.
"""

    raw_text = call_ollama(user_prompt)
    if not raw_text:
        return {
            "company_name": "",
            "position_applied": "",
            "application_date": "",
            "status": "applied",
        }

    text = extract_json_object(raw_text)

    for attempt in range(2):
        try:
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                text = re.sub(r",\s*([}\]])", r"\1", text)
            else:
                print("⚠️ Failed to parse JSON from model, got:", raw_text[:200])
                data = {
                    "company_name": "",
                    "position_applied": "",
                    "application_date": "",
                    "status": "applied",
                }

    company_name = str(data.get("company_name", "")).strip()
    position_applied = str(data.get("position_applied", "")).strip()
    application_date = str(data.get("application_date", "")).strip()
    status = str(data.get("status", "")).strip().lower()

    if status not in ["applied", "in progress", "rejected", "job offered"]:
        status = "applied"

    if len(position_applied) > 200:
        position_applied = position_applied[:200].rsplit(" ", 1)[0].strip()
    position_applied = position_applied.strip(" .|,-–—")

    return {
        "company_name": company_name,
        "position_applied": position_applied,
        "application_date": application_date,
        "status": status,
    }

def derive_application_date(date_received: str) -> str:
    if not date_received:
        return ""
    try:
        d = pd.to_datetime(date_received, errors="coerce")
    except Exception:
        return ""
    if pd.isna(d):
        return ""
    return d.strftime("%Y-%m-%d")


class LocalLLMJobParser:
    def __init__(self, input_filename: str, output_filename: str):
        code_dir = os.getcwd()
        project_root = os.path.dirname(code_dir)

        self.data_dir = os.path.join(project_root, "Data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.input_path = os.path.join(self.data_dir, input_filename)
        self.output_path = os.path.join(self.data_dir, output_filename)

    def run(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"{self.input_path} not found")

        # -----------------------------
        # 1) Load classified input file
        # -----------------------------
        df = pd.read_excel(self.input_path)

        # Filter to rows that are jobs (if job_label present)
        if "job_label" in df.columns:
            mask = df["job_label"].astype(str).str.lower().isin(["job", "1", "true", "yes"])
            df_jobs = df[mask].copy()
        else:
            df_jobs = df.copy()

        # Ensure required columns exist
        for col in ["subject", "body", "full_text", "date_received", "gmail_link"]:
            if col not in df_jobs.columns:
                df_jobs[col] = ""

        df_jobs["subject"] = df_jobs["subject"].fillna("").astype(str)
        df_jobs["body"] = df_jobs["body"].fillna("").astype(str)
        df_jobs["full_text"] = df_jobs["full_text"].fillna("").astype(str)
        df_jobs["date_received"] = df_jobs["date_received"].fillna("").astype(str)
        df_jobs["gmail_link"] = df_jobs["gmail_link"].fillna("").astype(str)

        # ----------------------------------------
        # 2) Load existing parsed file if present
        #    and detect already-processed mail_link
        # ----------------------------------------
        if os.path.exists(self.output_path):
            parsed_df_old = pd.read_excel(self.output_path)
            if "mail_link" in parsed_df_old.columns:
                parsed_df_old["mail_link"] = parsed_df_old["mail_link"].astype(str)
                seen_links = set(parsed_df_old["mail_link"].tolist())
            else:
                parsed_df_old = parsed_df_old.copy()
                parsed_df_old["mail_link"] = ""
                seen_links = set()
            print(f"[INFO] Loaded existing parsed file: {self.output_path}")
            print(f"[INFO] Already parsed rows: {len(parsed_df_old)}")
        else:
            parsed_df_old = None
            seen_links = set()
            print(f"[INFO] No existing parsed file found. Will create {self.output_path}")

        # ----------------------------------------
        # 3) Filter to ONLY new job emails to parse
        #    based on gmail_link / mail_link
        # ----------------------------------------
        df_jobs["gmail_link"] = df_jobs["gmail_link"].astype(str)
        df_new = df_jobs[~df_jobs["gmail_link"].isin(seen_links)].copy()

        if df_new.empty:
            print("[INFO] No new job emails to parse. Parsed file is up to date.")
            return parsed_df_old if parsed_df_old is not None else pd.DataFrame(
                columns=["mailcontent", "company_name", "position_applied",
                         "application_date", "status", "mail_link"]
            )

        print(f"[INFO] Total job emails in source: {len(df_jobs)}")
        print(f"[INFO] New job emails to parse this run: {len(df_new)}")

        # ----------------------------------------
        # 4) Run LLM parsing ONLY on new emails
        # ----------------------------------------
        mailcontents = []
        company_names = []
        positions = []
        app_dates = []
        statuses = []
        mail_links = df_new["gmail_link"].tolist()

        for i, (_, row) in enumerate(df_new.iterrows(), start=1):
            subject = row["subject"]
            body_full = row["full_text"] or row["body"]
            date_received = row["date_received"]

            full_original = (subject or "") + "\n" + (body_full or "")

            # Summary for mailcontent
            mailcontent = summarize_email(subject, body_full)
            mailcontents.append(full_original)

            # Heuristic position
            position_h = heuristic_position(subject, body_full)

            # LLM extraction
            info = call_llm_extract(subject, mailcontent, date_received)

            # Combine heuristic + LLM position
            raw_position = position_h if position_h else info["position_applied"]
            final_position = clean_final_position(raw_position)

            # Application date from received date
            app_date = derive_application_date(date_received)

            # Final status via rules + llm
            final_status = infer_status(full_original, info["status"])

            company_names.append(info["company_name"])
            positions.append(final_position)
            app_dates.append(app_date)
            statuses.append(final_status)

            if i % 10 == 0:
                print(f"Processed {i} new job emails...")

        result_new_df = pd.DataFrame({
            "mailcontent": mailcontents,
            "company_name": company_names,
            "position_applied": positions,
            "application_date": app_dates,
            "status": statuses,
            "mail_link": mail_links,
        })

        # ----------------------------------------
        # 5) Append to existing parsed file (if any)
        # ----------------------------------------
        if parsed_df_old is not None:
            final_df = pd.concat([parsed_df_old, result_new_df], ignore_index=True)
        else:
            final_df = result_new_df

        final_df.to_excel(self.output_path, index=False)
        print(f"[✓] Saved local-LLM parsed jobs to {self.output_path}")
        print(f"Total rows in parsed file now: {len(final_df)}")

        return final_df


def main():
    parser = LocalLLMJobParser(
        input_filename="mail_classified.xlsx",
        output_filename="mail_classified_llm_parsed.xlsx",
    )
    parser.run()

if __name__ == "__main__":
    main()

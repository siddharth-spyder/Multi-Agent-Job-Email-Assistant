# app.py
import os
import shutil
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st

# ---- import your existing modules (NO CHANGES to them) ----
import gmail_read
from gmail_read import GmailLiveReader, QUERY, OUTPUT_EXCEL  # reuse your constants
import predict
import ner
import rag  # this gives us rag.ask()

CODE_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_store")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


def _safe_read_excel(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            st.error(f"Error reading {path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def _last_modified(path: str) -> str:
    if not os.path.exists(path):
        return "No file yet"
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _flush_all() -> tuple[int, bool]:
    files = [
        os.path.join(DATA_DIR, "gmail_subject_body_date.xlsx"),
        os.path.join(DATA_DIR, "mail_classified.xlsx"),
        os.path.join(DATA_DIR, "mail_classified_llm_parsed.xlsx"),
    ]

    deleted_count = 0
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
                deleted_count += 1
            except Exception as e:
                st.error(f"Failed to delete {f}: {e}")

    chroma_deleted = False
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            chroma_deleted = True
            # recreate empty dir for future runs
            os.makedirs(CHROMA_DIR, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to delete Chroma store {CHROMA_DIR}: {e}")

    return deleted_count, chroma_deleted


def _save_uploaded_credentials(uploaded_file):
    """
    Save uploaded credentials.json next to app.py so GmailLiveReader can use it.
    """
    if uploaded_file is None:
        return

    creds_path = os.path.join(CODE_DIR, "credentials.json")
    try:
        with open(creds_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded credentials saved as `{creds_path}`")
    except Exception as e:
        st.error(f"Failed to save credentials.json: {e}")


def _run_custom_email_through_pipeline(subject: str, body: str, run_ner: bool = True):
    gmail_file = OUTPUT_EXCEL  # same as gmail_read uses
    classified_path = os.path.join(DATA_DIR, "mail_classified.xlsx")
    parsed_path = os.path.join(DATA_DIR, "mail_classified_llm_parsed.xlsx")

    # Unique ID + link for this custom email
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
    custom_id = f"custom_{timestamp_str}"
    custom_link = f"custom://{timestamp_str}"
    subj_val = subject or "(no subject)"
    body_val = body or ""

    # ---- Backup existing files (if they exist) ----
    had_gmail = os.path.exists(gmail_file)
    had_class = os.path.exists(classified_path)
    had_parsed = os.path.exists(parsed_path)

    gmail_backup = gmail_file + ".bak_custom" if had_gmail else None
    class_backup = classified_path + ".bak_custom" if had_class else None
    parsed_backup = parsed_path + ".bak_custom" if had_parsed else None

    try:
        if had_gmail:
            shutil.copy2(gmail_file, gmail_backup)
        if had_class:
            shutil.copy2(classified_path, class_backup)
        if had_parsed:
            shutil.copy2(parsed_path, parsed_backup)

        # ---- Create ONE-row Gmail-like DataFrame with dummy values ----
        now = datetime.now()
        custom_df = pd.DataFrame(
            [
                {
                    "id": custom_id,
                    "sender_name": "Custom User",
                    "sender_email": "custom@example.com",
                    "subject": subj_val,
                    "body": body_val,
                    "date_received": now,
                    "gmail_link": custom_link,
                }
            ]
        )

        # Overwrite gmail_subject_body_date.xlsx with ONLY this one row
        custom_df.to_excel(gmail_file, index=False)

        # ---- Run classifier (predict.main) ----
        predict.main()

        # Read classification result for this custom email
        class_row = None
        if os.path.exists(classified_path):
            cdf = pd.read_excel(classified_path)

            # Try several ways to find the row we just created
            mask = None
            if "id" in cdf.columns:
                mask = cdf["id"] == custom_id
            elif "gmail_link" in cdf.columns:
                mask = cdf["gmail_link"] == custom_link
            elif set(["subject", "body"]).issubset(cdf.columns):
                mask = (cdf["subject"] == subj_val) & (cdf["body"] == body_val)

            if mask is not None:
                class_row = cdf[mask]
            else:
                # last resort: if there's only one row, assume it's ours
                if len(cdf) == 1:
                    class_row = cdf

        # ---- If we want NER, force this row to be job_label='job' ----
        if run_ner:
            if os.path.exists(classified_path):
                cdf = pd.read_excel(classified_path)

                mask = None
                if "id" in cdf.columns:
                    mask = cdf["id"] == custom_id
                elif "gmail_link" in cdf.columns:
                    mask = cdf["gmail_link"] == custom_link
                elif set(["subject", "body"]).issubset(cdf.columns):
                    mask = (cdf["subject"] == subj_val) & (cdf["body"] == body_val)

                if mask is not None:
                    # Make sure ner.py won't skip this row
                    if "job_label" in cdf.columns:
                        cdf.loc[mask, "job_label"] = "job"
                    # write back so ner.py sees it
                    cdf.to_excel(classified_path, index=False)

            # Now run NER
            ner.main()

        # ---- Read NER result for this custom email (optional) ----
        ner_row = None
        if run_ner and os.path.exists(parsed_path):
            pdf = pd.read_excel(parsed_path)

            # Try matching by id, gmail_link, or mail_link
            mask_n = None
            if "id" in pdf.columns:
                mask_n = pdf["id"] == custom_id
            elif "gmail_link" in pdf.columns:
                mask_n = pdf["gmail_link"] == custom_link
            elif "mail_link" in pdf.columns:
                mask_n = pdf["mail_link"] == custom_link

            if mask_n is not None:
                ner_row = pdf[mask_n]

            # Fallback: if that still fails and there's only one row, assume it's ours
            if (ner_row is None or len(ner_row) == 0) and len(pdf) == 1:
                ner_row = pdf

        return class_row, ner_row

    except Exception as e:
        st.error(f"Error while running custom email through pipeline: {e}")
        return None, None

    finally:
        # ---- Restore backups / clean up ----
        # Restore if backup exists, else remove temp file we created
        if had_gmail and gmail_backup and os.path.exists(gmail_backup):
            shutil.move(gmail_backup, gmail_file)
        else:
            if not had_gmail and os.path.exists(gmail_file):
                os.remove(gmail_file)

        if had_class and class_backup and os.path.exists(class_backup):
            shutil.move(class_backup, classified_path)
        else:
            if not had_class and os.path.exists(classified_path):
                os.remove(classified_path)

        if had_parsed and parsed_backup and os.path.exists(parsed_backup):
            shutil.move(parsed_backup, parsed_path)
        else:
            if not had_parsed and os.path.exists(parsed_path):
                os.remove(parsed_path)


def fetch_new_emails_once(query: str = QUERY):

    st.write("Starting one-time Gmail fetch using GmailLiveReader...")
    st.write(f"Using Gmail query: `{query}`")

    reader = GmailLiveReader(
        credentials_path="credentials.json",  # same folder as app.py
        token_path="token.json",             # will be created here on first auth
        gmail_account_index=0,
    )

    # Load existing Excel if present
    if os.path.exists(OUTPUT_EXCEL):
        master_df = pd.read_excel(OUTPUT_EXCEL)
        if "id" in master_df.columns:
            seen_ids = set(master_df["id"].astype(str).tolist())
        else:
            seen_ids = set()
        st.write(f"Loaded {len(master_df)} rows from {OUTPUT_EXCEL}")
    else:
        master_df = pd.DataFrame(
            columns=[
                "id",
                "sender_name",
                "sender_email",
                "subject",
                "body",
                "date_received",
                "gmail_link",
            ]
        )
        seen_ids = set()
        st.write(f"No existing Excel found. Will create {OUTPUT_EXCEL}")

    # Fetch only NEW emails for the given query
    df_new = reader.fetch_new_as_dataframe(query, seen_ids)

    if df_new.empty:
        st.info("No new emails found for this query.")
        return master_df

    # Append and save
    master_df = pd.concat([master_df, df_new], ignore_index=True)
    master_df.to_excel(OUTPUT_EXCEL, index=False)

    st.success(f"Appended {len(df_new)} new email(s) to {OUTPUT_EXCEL}")
    st.subheader("Newly fetched emails for this query")
    st.dataframe(df_new[["date_received", "sender_email", "subject"]])

    return master_df


def run_classifier():
    """
    Call predict.main() exactly like your CLI.
    This reads gmail_subject_body_date.xlsx and writes mail_classified.xlsx.
    """
    st.write("Running classifier (predict.py)...")
    try:
        predict.main()
        st.success("Classification completed. mail_classified.xlsx updated.")
    except Exception as e:
        st.error(f"Error running classifier: {e}")


def run_ner():
    """
    Call ner.main() exactly like your CLI.
    This reads mail_classified.xlsx and writes mail_classified_llm_parsed.xlsx.
    """
    st.write("Running NER parser (ner.py)...")
    try:
        ner.main()
        st.success("NER completed. mail_classified_llm_parsed.xlsx updated.")
    except Exception as e:
        st.error(f"Error running NER: {e}")


# ========== PAGES ==========

def page_fetch_and_classify():
    st.header("1️⃣ Gmail Fetch + Job Classification")

    st.markdown("This page uses your existing **gmail_read.py** + **predict.py** logic.")

    # --- date range selector for Gmail query (FROM - TILL) ---
    st.subheader("Gmail Fetch Settings")
    col_from, col_to, col_info = st.columns([1, 1, 2])

    today = date.today()
    default_from = today.replace(day=1)

    with col_from:
        from_date: date = st.date_input(
            "Fetch emails FROM (inclusive)",
            value=default_from,
            help="We'll build a Gmail query with `after:` using this date.",
        )
    with col_to:
        to_date: date = st.date_input(
            "Fetch emails TILL (inclusive)",
            value=today,
            help="We'll use `before:` as (TILL + 1 day) so this date is included.",
        )
    with col_info:
        st.caption(
            "Emails between these two dates will be considered when you click "
            "**Fetch NEW emails** (using Gmail `after:` and `before:`)."
        )

    # Build dynamic query: after:FROM before:(TILL+1)
    from_str = from_date.strftime("%Y/%m/%d")
    before_date = to_date + timedelta(days=1)
    before_str = before_date.strftime("%Y/%m/%d")
    dynamic_query = f"after:{from_str} before:{before_str}"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Fetch NEW emails from Gmail (one batch)"):
            fetch_new_emails_once(query=dynamic_query)

    with col_btn2:
        if st.button("Run job/non-job classifier on emails"):
            run_classifier()

    st.markdown("---")
    st.subheader("Raw Gmail Export (All stored emails)")

    # Show whatever OUTPUT_EXCEL points to (same file pipeline uses)
    gmail_df = _safe_read_excel(OUTPUT_EXCEL)
    st.caption(f"File: `{OUTPUT_EXCEL}`  •  Last updated: {_last_modified(OUTPUT_EXCEL)}")
    if gmail_df.empty:
        st.info("No Gmail Excel yet or file is empty.")
    else:
        # Optional: filter view by date range chosen above
        if "date_received" in gmail_df.columns:
            try:
                gmail_df["date_received"] = pd.to_datetime(gmail_df["date_received"])
                mask = (gmail_df["date_received"].dt.date >= from_date) & (
                    gmail_df["date_received"].dt.date <= to_date
                )
                filtered = gmail_df[mask]
                st.write(
                    f"Showing emails between **{from_date}** and **{to_date}** "
                    f"({len(filtered)} rows)."
                )
                st.dataframe(filtered.head(300))
            except Exception:
                st.dataframe(gmail_df.head(300))
        else:
            st.dataframe(gmail_df.head(300))

    st.markdown("---")
    st.subheader("Classified Emails (mail_classified.xlsx)")
    classified_path = os.path.join(DATA_DIR, "mail_classified.xlsx")
    class_df = _safe_read_excel(classified_path)
    st.caption(f"Path: `{classified_path}`  •  Last updated: {_last_modified(classified_path)}")
    if class_df.empty:
        st.info("No classified file yet. Run the classifier first.")
    else:
        # small summary: job vs non_job counts
        if "job_label" in class_df.columns:
            counts = class_df["job_label"].value_counts()
            st.write("Job vs Non-Job counts:")
            st.bar_chart(counts)

        cols_to_show = [
            "date_received",
            "sender_email",
            "subject",
            "job_label",
            "prob_job",
        ]
        cols_to_show = [c for c in cols_to_show if c in class_df.columns]

        # --- show two separate tables: JOB vs NON-JOB ---
        if "job_label" in class_df.columns:
            job_df = class_df[class_df["job_label"] == "job"]
            non_job_df = class_df[class_df["job_label"] != "job"]

            col_job, col_non = st.columns(2)

            with col_job:
                st.markdown("### ✅ Classified as **JOB**")
                if job_df.empty:
                    st.info("No emails classified as job yet.")
                else:
                    st.dataframe(job_df[cols_to_show].head(300))

            with col_non:
                st.markdown("### 📩 Classified as **NON-JOB / OTHER**")
                if non_job_df.empty:
                    st.info("No emails classified as non-job.")
                else:
                    st.dataframe(non_job_df[cols_to_show].head(300))
        else:
            st.dataframe(class_df[cols_to_show].head(300))

    # ---- danger zone: flush button (still here for convenience) ----
    st.markdown("---")
    st.subheader("Danger Zone: Reset Stored Data (same as top button)")

    with st.expander("⚠️ Flush all data & Chroma store"):
        st.warning(
            "This will delete:\n\n"
            "- `Data/gmail_subject_body_date.xlsx`\n"
            "- `Data/mail_classified.xlsx`\n"
            "- `Data/mail_classified_llm_parsed.xlsx`\n"
            "- Entire `chroma_store/` directory\n\n"
            "Use this if you want to restart from scratch."
        )
        confirm = st.checkbox("I understand, delete all of the above.")
        if st.button("Flush data & Chroma", disabled=not confirm, key="flush_bottom"):
            n_files, chroma_deleted = _flush_all()
            msg = f"Deleted {n_files} data file(s)."
            if chroma_deleted:
                msg += " Chroma store cleared."
            st.success(msg)


def page_ner_view():
    st.header("2️⃣ Job Entity Extraction (NER)")

    st.markdown("This page uses your **ner.py** to parse job emails into structured fields.")

    if st.button("Run NER on classified job emails"):
        run_ner()

    st.markdown("---")
    st.subheader("Parsed Job Records (mail_classified_llm_parsed.xlsx)")

    parsed_path = os.path.join(DATA_DIR, "mail_classified_llm_parsed.xlsx")
    parsed_df = _safe_read_excel(parsed_path)
    st.caption(f"Path: `{parsed_path}`  •  Last updated: {_last_modified(parsed_path)}")

    if parsed_df.empty:
        st.info("No parsed job records yet. Run NER first.")
    else:
        st.write(f"Total parsed rows: {len(parsed_df)}")
        cols = [
            "company_name",
            "position_applied",
            "application_date",
            "status",
            "mail_link",
        ]
        show_cols = [c for c in cols if c in parsed_df.columns]
        st.dataframe(parsed_df[show_cols].head(300))


def page_rag():
    st.header("3️⃣ RAG Assistant – Ask About Your Applications")

    st.markdown(
        """
This uses your **rag.py**:

- Loads `mail_classified_llm_parsed.xlsx`
- Uses ChromaDB + embeddings
- Calls `rag.ask(question)` with your local Ollama Llama 3.1
        """
    )

    question = st.text_area(
        "Ask something like:",
        "What is the latest status of my ASML application?",
        height=80,
    )

    if st.button("Ask RAG"):
        if not question.strip():
            st.warning("Type a question first.")
            return

        try:
            with st.spinner("Thinking..."):
                answer = rag.ask(question)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(
                "RAG failed. Make sure `mail_classified_llm_parsed.xlsx` exists "
                "and Chroma store is set up. Error:\n\n" + str(e)
            )


def page_custom_email():
    """
    4️⃣ Custom tab: user types subject + body and we show:
      - Job / non-job classification  (Button 1)
      - NER extraction ONLY           (Button 2)
    Both run through real Excel + predict.main() + ner.main() pipeline.
    """
    st.header("4️⃣ Custom Email Test – Classification + NER")

    st.markdown(
        "Use this tab to test your model on a **manual email** "
        "(subject + body) without going through Gmail fetch."
    )

    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body", height=200)

    # ---------- BUTTON ROW ----------
    col1, col2 = st.columns(2)

    # BUTTON 1 — CLASSIFICATION ONLY
    with col1:
        classify_clicked = st.button("🔍 Classify Only")

    # BUTTON 2 — NER ONLY
    with col2:
        ner_clicked = st.button("🧠 NER Only")

    # ---------- CLASSIFICATION ----------
    if classify_clicked:
        if not subject.strip() and not body.strip():
            st.warning("Please enter at least a subject or body.")
            return

        with st.spinner("Running classification (predict.main())..."):
            class_row, _ = _run_custom_email_through_pipeline(
                subject=subject,
                body=body,
                run_ner=False,   # only classification
            )

        st.markdown("---")
        st.subheader("🔍 Classification Output")

        if class_row is None or len(class_row) == 0:
            st.info("No classification found for this custom email.")
        else:
            st.dataframe(class_row)

    # ---------- NER EXECUTION ----------
    if ner_clicked:
        if not subject.strip() and not body.strip():
            st.warning("Please enter at least a subject or body.")
            return

        with st.spinner("Running NER (predict.main() + ner.main())..."):
            # NER requires classification first
            class_row, ner_row = _run_custom_email_through_pipeline(
                subject=subject,
                body=body,
                run_ner=True,
            )

        st.markdown("---")
        st.subheader("🧠 NER Output")

        if ner_row is None or len(ner_row) == 0:
            st.info("No NER entities found for this custom email.")
        else:
            st.dataframe(ner_row)


# ========== main ==========

def main():
    st.set_page_config(page_title="Job Email Tracker (Existing Pipeline)", layout="wide")

    st.title("Job Email Tracker – Streamlit Wrapper")
    st.caption(
        "This UI orchestrates your existing gmail_read.py, predict.py, ner.py, and rag.py. "
        "It does not change their logic."
    )

    # ========== TOP BAR: Flush + Credentials Upload ==========
    st.markdown("### ⚙️ Global Settings")

    col_top_left, col_top_mid, col_top_right = st.columns([1.5, 2, 2])

    with col_top_left:
        st.markdown("**Danger Zone – Flush Everything**")
        if st.button(" Flush data & Chroma (TOP)", key="flush_top"):
            n_files, chroma_deleted = _flush_all()
            msg = f"Deleted {n_files} data file(s)."
            if chroma_deleted:
                msg += " Chroma store cleared."
            st.success(msg)

    with col_top_mid:
        st.markdown("**Upload your `credentials.json`**")
        uploaded_creds = st.file_uploader(
            "Drop your Google API credentials.json here",
            type=["json"],
            key="creds_uploader",
            help="This will be saved as `credentials.json` next to app.py "
                 "and used by GmailLiveReader.",
        )
        if uploaded_creds is not None:
            _save_uploaded_credentials(uploaded_creds)

    with col_top_right:
        st.markdown("**Info**")
        st.write(
            "- `gmail_read.py` will use `credentials.json` + `token.json` in this folder.\n"
            "- Each user can upload their own credentials without touching your file."
        )

    st.markdown("---")

    # ========== TABS ==========
    tab1, tab2, tab3, tab4 = st.tabs(
        ["1. Fetch + Classify", "2. NER Parsed Jobs", "3. RAG Assistant", "4. Custom Email Test"]
    )

    with tab1:
        page_fetch_and_classify()
    with tab2:
        page_ner_view()
    with tab3:
        page_rag()
    with tab4:
        page_custom_email()


if __name__ == "__main__":
    main()

from __future__ import print_function
import os
import base64
import time
import re
import pandas as pd
from email import message_from_bytes
from email.utils import parseaddr
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup


DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


QUERY = "after:2025/10/28"


INTERVAL = 60  


code_dir = os.getcwd()
project_root = os.path.dirname(code_dir)
data_dir = os.path.join(project_root, "Data")
os.makedirs(data_dir, exist_ok=True)

OUTPUT_EXCEL = os.path.join(data_dir, "gmail_subject_body_date.xlsx")



_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_for_excel(value, max_len=32000):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = _ILLEGAL_XML_RE.sub("", value)
    if len(value) > max_len:
        value = value[:max_len]
    return value


class GmailLiveReader:
    def __init__(
        self,
        credentials_path="credentials.json",
        token_path="token.json",
        gmail_account_index=0,
    ):
        self.credentials_path = os.path.abspath(credentials_path)
        self.token_path = os.path.abspath(token_path)
        self.gmail_account_index = gmail_account_index
        self.gmail_web_base = f"https://mail.google.com/mail/u/{gmail_account_index}/#all/"
        self.scopes = DEFAULT_SCOPES

        self.service = self._authenticate()


    def _authenticate(self):
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)


    @staticmethod
    def html_to_text(html):
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ", strip=True).split())

    def list_ids(self, query):
        """
        Return a list of *all* message IDs matching the query.
        Handles pagination via nextPageToken.
        """
        ids = []
        page_token = None

        while True:
            response = self.service.users().messages().list(
                userId="me",
                q=query,
                maxResults=500,        
                pageToken=page_token,    
                includeSpamTrash=True,
            ).execute()

            ids.extend([m["id"] for m in response.get("messages", [])])

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        print(f"[DEBUG] list_ids → found {len(ids)} messages for query: {query}")
        return ids

    def get_details(self, msg_id):
        msg = self.service.users().messages().get(
            userId="me", id=msg_id, format="raw"
        ).execute()

        raw_msg = base64.urlsafe_b64decode(msg["raw"])
        email_msg = message_from_bytes(raw_msg)

        sender_name, sender_email = parseaddr(email_msg.get("From", ""))
        subject = email_msg.get("Subject", "")
        date_raw = email_msg.get("Date", "")


        date_str = date_raw
        try:
            parsed = datetime.strptime(date_raw[:31], "%a, %d %b %Y %H:%M:%S %z")
            date_str = parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        plain, html = "", ""

        if email_msg.is_multipart():
            for part in email_msg.walk():
                ctype = part.get_content_type()
                dispo = str(part.get("Content-Disposition") or "")
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)

                if not payload:
                    continue

                if ctype == "text/plain" and "attachment" not in dispo:
                    plain += payload.decode(charset, errors="ignore")
                elif ctype == "text/html" and "attachment" not in dispo:
                    html += payload.decode(charset, errors="ignore")
        else:
            ctype = email_msg.get_content_type()
            charset = email_msg.get_content_charset() or "utf-8"
            payload = email_msg.get_payload(decode=True)
            if payload:
                if ctype == "text/plain":
                    plain = payload.decode(charset, errors="ignore")
                else:
                    html = payload.decode(charset, errors="ignore")

        body = plain if plain.strip() else self.html_to_text(html)


        return {
            "id": clean_for_excel(msg_id),
            "sender_name": clean_for_excel(sender_name),
            "sender_email": clean_for_excel(sender_email),
            "subject": clean_for_excel(subject),
            "body": clean_for_excel(body),
            "date_received": clean_for_excel(date_str),
            "gmail_link": clean_for_excel(f"{self.gmail_web_base}{msg_id}"),
        }

    def fetch_new_as_dataframe(self, query, seen_ids):
        """
        - query: Gmail search string
        - seen_ids: set of already-processed message IDs (mutated in place)
        Returns a DataFrame with ONLY new emails (ids not in seen_ids).
        """
        all_ids = self.list_ids(query)
        new_ids = [mid for mid in all_ids if mid not in seen_ids]

        if not new_ids:
            return pd.DataFrame(columns=[
                "id", "sender_name", "sender_email",
                "subject", "body", "date_received", "gmail_link"
            ])

        rows = []
        for mid in new_ids:
            details = self.get_details(mid)
            rows.append(details)
            seen_ids.add(mid)

        df = pd.DataFrame(rows)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].map(clean_for_excel)

        return df



if __name__ == "__main__":
    reader = GmailLiveReader(
        credentials_path="credentials.json",
        token_path="token.json",
        gmail_account_index=0,
    )

    
    if os.path.exists(OUTPUT_EXCEL):
        master_df = pd.read_excel(OUTPUT_EXCEL)
        for col in master_df.select_dtypes(include=["object"]).columns:
            master_df[col] = master_df[col].map(clean_for_excel)

        if "id" in master_df.columns:
            seen_ids = set(master_df["id"].astype(str).tolist())
        else:
            seen_ids = set()
        print(f"[INIT] Loaded {len(master_df)} rows from {OUTPUT_EXCEL}")
    else:
        master_df = pd.DataFrame(columns=[
            "id", "sender_name", "sender_email",
            "subject", "body", "date_received", "gmail_link"
        ])
        seen_ids = set()
        print(f"[INIT] No existing Excel found. Will create {OUTPUT_EXCEL}")

    print("Live Gmail Reader (NEW EMAILS ONLY, APPENDING TO EXCEL)")
    print(f"Query: {QUERY}")
    print(f"Interval: {INTERVAL} seconds")
    print(f"Output Excel: {OUTPUT_EXCEL}")
    print("Press Ctrl+C to stop.\n")

    while True:
        loop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df_new = reader.fetch_new_as_dataframe(QUERY, seen_ids)

        if df_new.empty:
            print(f"[{loop_time}] No new emails.")
        else:
            master_df = pd.concat([master_df, df_new], ignore_index=True)

            for col in master_df.select_dtypes(include=["object"]).columns:
                master_df[col] = master_df[col].map(clean_for_excel)

            master_df.to_excel(OUTPUT_EXCEL, index=False)

            print(f"[{loop_time}] {len(df_new)} new email(s) appended to {OUTPUT_EXCEL}")
            print(df_new[["date_received", "sender_email", "subject"]])
            print()

        print("-" * 80)
        time.sleep(INTERVAL)

# flush.py
import os
import shutil
import glob
import chromadb


code_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(code_dir)                

DATA_DIR = os.path.join(project_root, "Data")
CHROMA_DIR = os.path.join(code_dir, "chroma_store")


EXCEL_PATTERNS = [
    "mail_classified.xlsx",
    "mail_classified*.xlsx",
    "mail_classified_llm_parsed.xlsx",
    "gmail_subject_body_date.xlsx",
]

def flush_chroma_folder():
    if os.path.exists(CHROMA_DIR):
        print(f"Deleting Chroma store at: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
        print("✔ Chroma folder deleted.\n")
    else:
        print(f"No Chroma store found at: {CHROMA_DIR}\n")


def flush_chroma_collection(collection_name: str = "job_email_collection"):
    if not os.path.exists(CHROMA_DIR):
        print(f"No Chroma directory at {CHROMA_DIR}, nothing to delete.")
        return

    print(f"Connecting to Chroma at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(name=collection_name)
        print(f"✔ Deleted collection '{collection_name}' from Chroma.\n")
    except Exception as e:
        print(f"Could not delete collection '{collection_name}': {e}\n")


def flush_data_files():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    print(f"Looking for derived files in: {DATA_DIR}")

    deleted_any = False
    for pattern in EXCEL_PATTERNS:
        full_pattern = os.path.join(DATA_DIR, pattern)
        for path in glob.glob(full_pattern):
            try:
                print(f"Deleting file: {path}")
                os.remove(path)
                deleted_any = True
            except Exception as e:
                print(f"Error deleting {path}: {e}")

    if not deleted_any:
        print("No matching derived Excel files found to delete.\n")
    else:
        print("✔ Finished deleting derived Excel files.\n")



# Main
if __name__ == "__main__":
    print("\n=== FLUSH JOB EMAIL DATA & VECTOR STORE ===\n")
    flush_data_files()

    print("All done.\n")

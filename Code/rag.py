import os
from typing import TypedDict, List, Any, Optional

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END


# Path Setup

code_dir = os.path.dirname(os.path.abspath(__file__))   
project_root = os.path.dirname(code_dir)                

DATA_DIR = os.path.join(project_root, "Data")
os.makedirs(DATA_DIR, exist_ok=True)

CHROMA_DIR = os.path.join(project_root, "chroma_store")
os.makedirs(CHROMA_DIR, exist_ok=True)

EXCEL_FILENAME = "mail_classified_llm_parsed.xlsx"
EXCEL_PATH = os.path.join(DATA_DIR, EXCEL_FILENAME)




class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Any]
    answer: str


_rag_app = None      
_retriever = None    
_df_parsed: Optional[pd.DataFrame] = None  

# Prompt Template

prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are my personal job-application tracking assistant.

You are given CONTEXT containing several emails about my job applications.
Each email is represented with STRUCTURED FIELDS:

- company_name
- position_applied
- application_date  (YYYY-MM-DD, when I received that email)
- status            (one of: applied, in progress, rejected, job offered)
- mail_link         (direct Gmail link)
- summary           (short natural language summary of the email, from the mail body)

YOUR RULES:
1. Always base your answer primarily on these structured fields and the summary,
   especially "company_name", "position_applied", "application_date", and "status".
2. Directly answer the user's QUESTION. Do NOT invent your own questions.
   Do NOT output Q&A-style lists unless the user explicitly asks for that format.
3. When the question mentions a company or role, match them
   case-insensitively against company_name and position_applied.
4. If multiple emails correspond to the same application
   (same company_name + position_applied), assume the record
   with the LATEST application_date is the current status.
5. If the question is aggregate (e.g. "How many are in progress?"),
   reason over the provided CONTEXT and count by status.
6. Ignore pure newsletters or generic marketing emails if they do not
   describe an application, decision, or clear change of status.
7. If you don't find anything relevant, say so clearly.
8. Do NOT invent companies, roles, dates, or statuses that are not in the CONTEXT.

QUESTION:
{question}

CONTEXT:
{context}

Now answer concisely and factually. If useful, you may mention the mail_link(s)
so I can open the exact email.
"""
)


def _maybe_answer_with_analytics(question: str) -> Optional[str]:
    """
    For questions like "total applications count", "how many rejections",
    "overall pipeline stats", "companywise insights", etc., we don't want
    to rely on just top-k retrieved docs. We use the full _df_parsed DataFrame.

    Returns:
        - A natural language answer if we detect an aggregate/stats question.
        - None otherwise (so we fall back to normal RAG).
    """
    global _df_parsed

    if _df_parsed is None:
        return None

    q = question.strip()
    if not q:
        return None

    q_lower = q.lower()

   
    trigger_words = [
        "how many",
        "count",
        "total",
        "overall",
        "summary",
        "breakdown",
        "pipeline",
        "statistics",
        "stats",
        "overview",
        "insight",
        "insights",
        "companywise",
        "company-wise",
        "company wise",
        "per company",
        "by company",
    ]
    if not any(t in q_lower for t in trigger_words):
        return None

    df = _df_parsed.copy()

    # Normalize key columns
    df["company_name"] = df["company_name"].fillna("").astype(str).str.strip()
    df["position_applied"] = df["position_applied"].fillna("").astype(str).str.strip()
    df["status"] = df["status"].fillna("").astype(str).str.lower().str.strip()

    companies = sorted(
        c for c in df["company_name"].unique() if isinstance(c, str) and c.strip()
    )

    matched_companies = []
    q_lower_spaced = f" {q_lower} "
    for c in companies:
        c_lower = c.lower()
        if c_lower and c_lower in q_lower_spaced:
            matched_companies.append(c)

    if matched_companies:
        df_q = df[df["company_name"].isin(matched_companies)].copy()
        scope_text = f"for {', '.join(matched_companies)}"
    else:
        df_q = df
        scope_text = "across all companies"

    if df_q.empty:
        return f"I couldn't find any job-related emails {scope_text} in your parsed data."


    total_emails = len(df_q)

    df_apps = df_q[
        (df_q["company_name"] != "") & (df_q["position_applied"] != "")
    ].copy()
    unique_apps_df = df_apps[["company_name", "position_applied"]].drop_duplicates()
    num_unique_apps = len(unique_apps_df)

    base_statuses = ["applied", "in progress", "rejected", "job offered"]
    status_counts = {s: int((df_q["status"] == s).sum()) for s in base_statuses}

    other_mask = ~df_q["status"].isin(base_statuses) & (df_q["status"] != "")
    other_count = int(other_mask.sum())
    if other_count > 0:
        status_counts["other"] = other_count

    lines = []
    lines.append(f"I found **{total_emails}** job-related emails {scope_text}.")
    lines.append(f"These correspond to about **{num_unique_apps}** unique applications (by company + position).")

    lines.append("\nStatus breakdown (by email):")
    for s in base_statuses:
        lines.append(f"- {s}: {status_counts[s]}")
    if "other" in status_counts:
        lines.append(f"- other/unknown: {status_counts['other']}")


    if any(t in q_lower for t in ["companywise", "company-wise", "company wise", "per company", "by company", "insight", "insights"]):
        if not df_apps.empty:
            lines.append("\nCompany-wise application insights (distinct applications):")
            
            grouped = df_apps.groupby("company_name")["status"].value_counts().unstack(fill_value=0)
            
            grouped["__total__"] = grouped.sum(axis=1)
            grouped = grouped.sort_values("__total__", ascending=False).drop(columns="__total__")
            
            max_companies = 15
            for i, (comp, row) in enumerate(grouped.iterrows()):
                if i >= max_companies:
                    lines.append(f"...and more companies not shown here.")
                    break
                total_comp = int(row.sum())
                parts = [f"{s}: {int(row.get(s, 0))}" for s in base_statuses if row.get(s, 0) > 0]
                parts_str = ", ".join(parts) if parts else "no clear status"
                lines.append(f"- {comp}: {total_comp} applications ({parts_str})")

    if "total application" in q_lower or "total applications" in q_lower:
        lines.append(
            f"\nSo, your **total number of distinct applications** {scope_text} "
            f"is approximately **{num_unique_apps}**."
        )

    return "\n".join(lines)



def _init_rag_app():
    """
    Lazy initializer:
    - Loads mail_classified_llm_parsed.xlsx into _df_parsed
    - Incrementally updates Chroma (only new docs based on mail_link/doc_id)
    - Builds LangGraph pipeline

    Sets global _rag_app, _retriever, and _df_parsed.
    """
    global _rag_app, _retriever, _df_parsed


    if _rag_app is not None:
        return

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"RAG Excel not found at {EXCEL_PATH}. "
            "Run classification + NER pipeline first to generate mail_classified_llm_parsed.xlsx."
        )

    print(f">>> Loading parsed job emails from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    required_cols = [
        "mailcontent",
        "company_name",
        "position_applied",
        "application_date",
        "status",
        "mail_link",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    _df_parsed = df.copy()

    df["mail_link"] = df["mail_link"].fillna("").astype(str)
    df["doc_id"] = df["mail_link"]


    mask_empty = df["doc_id"] == ""
    df.loc[mask_empty, "doc_id"] = [f"row_{i}" for i in df.index[mask_empty]]


    print(f">>> Initializing Chroma vector database at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name="job_email_collection",
        metadata={"hnsw:space": "cosine"},
    )


    existing = collection.get(limit=1_000_000)   
    existing_ids = set(existing.get("ids", []))

    print(f">>> Existing vectors in Chroma: {len(existing_ids)}")

    df_new = df[~df["doc_id"].isin(existing_ids)].copy()

    if df_new.empty:
        print(">>> No new rows to embed. Chroma store is up to date.\n")
    else:
        print(f">>> New rows to embed: {len(df_new)}")

        ids = df_new["doc_id"].tolist()
        documents = df_new["mailcontent"].astype(str).tolist()
        metadatas = df_new[
            ["company_name", "position_applied", "application_date", "status", "mail_link"]
        ].astype(str).to_dict(orient="records")

        print("\n>>> Loading embedding model (BAAI/bge-large-en-v1.5)...")
        embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

        print(">>> Embedding email texts (new only)...")
        embeddings = embedder.encode(documents, convert_to_numpy=True)

        print(">>> Storing new embeddings in Chroma...")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print("✔ Embedding + storage for new rows complete.\n")



    lc_embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    vectorstore = Chroma(
        client=client,
        collection_name="job_email_collection",
        embedding_function=lc_embedder,
        persist_directory=CHROMA_DIR,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    _retriever = retriever

    print(">>> Using Ollama Llama 3.1 model...")

    llm = Ollama(
        model="llama3.1",   
        temperature=0.2,
    )


    def retrieve_node(state: RAGState):
        """Retrieve top matching email records."""
        docs = retriever.invoke(state["question"])
        state["retrieved_docs"] = docs
        return state

    def llm_node(state: RAGState):
        """Generate a factual answer using retrieved emails."""
        if not state["retrieved_docs"]:
            state["answer"] = (
                "I couldn't find any matching job application emails for this question."
            )
            return state


        context_lines = []
        for i, doc in enumerate(state["retrieved_docs"], start=1):
            meta = doc.metadata or {}
            context_lines.append(
                f"EMAIL {i}:\n"
                f"  company_name: {meta.get('company_name', '')}\n"
                f"  position_applied: {meta.get('position_applied', '')}\n"
                f"  application_date: {meta.get('application_date', '')}\n"
                f"  status: {meta.get('status', '')}\n"
                f"  mail_link: {meta.get('mail_link', '')}\n"
                f"  summary: {doc.page_content}\n"
            )
        context = "\n".join(context_lines)

        final_prompt = prompt_template.format(
            question=state["question"],
            context=context,
        )

        output = llm.invoke(final_prompt)

        if isinstance(output, str):
            state["answer"] = output.strip()
        else:
            try:
                state["answer"] = output.content.strip()
            except Exception:
                state["answer"] = str(output).strip()

        return state


    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", llm_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    _rag_app = app

    print(">>> RAG app initialized.")



def ask(question: str) -> str:
    """
    Public function used by Streamlit (app.py) and CLI.

    - Lazily initializes the RAG pipeline on first call.
    - FIRST tries to answer aggregate/statistics questions using the FULL
      DataFrame (_df_parsed), so counts and totals are correct.
    - Otherwise, falls back to RAG (retriever + LLM).
    """
    global _rag_app

    if not question or not str(question).strip():
        return "Please provide a non-empty question."


    if _rag_app is None:
        _init_rag_app()

    analytics_answer = _maybe_answer_with_analytics(question)
    if analytics_answer is not None:
        return analytics_answer


    initial_state: RAGState = {
        "question": question,
        "retrieved_docs": [],
        "answer": "",
    }
    final_state = _rag_app.invoke(initial_state)
    return final_state["answer"]


if __name__ == "__main__":
    print("\n Job Application RAG System Ready.\n")
    print(f"(Using data from: {EXCEL_PATH})")
    print(f"(Chroma store at: {CHROMA_DIR})\n")

    while True:
        try:
            q = input("Ask about your job applications → ")
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower().strip() in ["exit", "quit"]:
            break

        ans = ask(q)
        print("\n--- ANSWER ---")
        print(ans)
        print("--------------\n")

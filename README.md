# Multi Agent Gmail Job Application Tracker
#### Automated Email Classification • Job Entity Extraction • RAG Search • ChromaDB Storage • Streamlit Demo

## DATS 6312 – Introduction to Natural Language Processing  
George Washington University  
Instructor: Dr. Amir Jafari  
Group Members: Aswin Balaji Thippa Ramesh, Siddharth Sriram, Rahul Arvind

# Project Overview

This repository contains the full implementation of our final NLP project for DATS 6312 at The George Washington University.  
The project focuses on designing and deploying an automated, end-to-end system that:

- Connects securely to the user’s Gmail inbox  
- Classifies incoming emails as job-related or non-job-related  
- Extracts structured metadata such as company name, position applied for, status, application date, and original Gmail link  
- Stores processed emails as embeddings inside a vector database (ChromaDB)  
- Enables a Retrieval-Augmented Generation (RAG) interface to query job application history using natural language  
- Provides a Streamlit-based demo application that runs the full pipeline

This system replicates a real-world NLP workflow and uses ML models, NER, embedding search, vector databases, and LLM-based retrieval to create a personal job-application assistant.
A **Streamlit demo app (`app.py`)** is included to run everything visually.

---

## Features

- ✔ Gmail API email fetching  
- ✔ Job vs non-job classification (Classifier 1 performs best)  
- ✔ NER extraction (Company, Position, Date, Status, Link)  
- ✔ RAG + ChromaDB storage  
- ✔ Chat with your processed job emails  
- ✔ End-to-end Streamlit UI  

---

# Installation & Setup

## 1) Clone the Repository

```bash
git clone https://github.com/<your-username>/Multi-Agent-Gmail-Job-Application-Tracker.git
cd Multi-Agent-Gmail-Job-Application-Tracker/Code
```

---

## 2) Install Requirements

Before running anything:

```bash
pip install -r requirements.txt
```

 **IMPORTANT:**  
Make sure you have a **local LLaMA model downloaded** (Ollama recommended)

Example:

```bash
ollama pull llama3
```

This is required for the RAG chat and some classifier logic.

---

## 3) Set Up Gmail API (Google Cloud Console)

1. Go to: https://console.cloud.google.com  
2. Create a **new project**  
3. Enable **Gmail API**  
4. Go to **OAuth Consent Screen**
   - Select **External**
   - Add yourself as a **trusted user**
5. Go to **Credentials → Create Credentials → OAuth Client ID**
6. Choose **Desktop App**
7. Download `credentials.json`
8. Place it in:

```
NLP-Project/Code/credentials.json
```

---

# Running the Pipeline

## STEP 1 — Train the Classifier (Classifier 1)

```bash
cd "Classifier 1"
python training.py
```

Generates:

```
best_classifier.pkl
```

---

## STEP 2 — Fetch Gmail Emails

```bash
cd ..
python gmail_read.py
```

Creates:

```
Data/gmail_subject_body_date.xlsx
```

---

## STEP 3 — Classify Emails (job / non-job)

```bash
python predict.py
```

Outputs:

```
Data/mail_classified.xlsx
```

---

## STEP 4 — Extract Entities (NER)

```bash
python ner.py
```

Outputs:

```
Data/mail_classified_llm_parsed.xlsx
```

---

## STEP 5 — Store in ChromaDB (RAG)

```bash
python rag.py
```

This will:

- Embed emails using BGE or MiniLM  
- Create/update `chroma_store/`  
- Prepare data for semantic search + chat

---

## STEP 6 — Run Streamlit Demo App

For a complete visual demo:

```bash
streamlit run app.py
```

This will open a full UI to:
- Fetch emails  
- Classify job vs non-job  
- Extract structured job info  
- Store into vector DB  
- Chat with your job emails  

---

## Example Use Cases

“Show me all rejected applications”

“When did I apply to Deloitte?”

“How many companies responded last month?”

“Summarize all my job applications so far”

<div align="center">

# 🩺 DermaCam Chatbot

### AI-Powered Skin Disease Assistant — English & العربية

*Graduation Project · Faculty of Computer Science*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-F55036?style=flat-square)](https://groq.com)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0095D5?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

> **DermaCam** is an intelligent, bilingual chatbot that answers questions about skin diseases using Retrieval-Augmented Generation (RAG). It combines a curated medical knowledge base with a state-of-the-art language model to deliver accurate, context-grounded responses — while clearly staying within its defined medical scope.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Supported Diseases](#-supported-diseases)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Setup & Installation](#-setup--installation)
- [Running the Chatbot](#-running-the-chatbot)
- [Features](#-features)
- [Confidence Scoring](#-confidence-scoring)
- [Chatbot Boundaries](#-chatbot-boundaries)
- [Dependencies](#-dependencies)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

DermaCam is built as the conversational AI component of a larger skin disease diagnosis system. It does **not** diagnose — it educates. The chatbot answers questions about 8 skin conditions in both **English** and **Arabic** (including Egyptian colloquial dialect), detects the user's language automatically, and always stays within its defined medical scope.

The system is powered by a **RAG pipeline**: instead of relying solely on the language model's general knowledge, every response is grounded in a hand-crafted medical knowledge base that is retrieved at query time using semantic vector search.

---

## 🦠 Supported Diseases

| # | English Name | الاسم بالعربية | Severity |
|---|-------------|----------------|----------|
| 1 | Acne and Rosacea | حب الشباب والوردية | 🟡 Low |
| 2 | Actinic Keratosis | التقران الشعاعي | 🔴 High |
| 3 | Chickenpox | جدري الماء | 🟡 Moderate |
| 4 | Eczema (Atopic Dermatitis) | الأكزيما | 🟡 Moderate |
| 5 | Monkeypox | جدري القردة | 🟠 Moderate–High |
| 6 | Nail Fungus (Onychomycosis) | فطريات الأظافر | 🟡 Low |
| 7 | Skin Cancer | سرطان الجلد | 🔴 High |
| 8 | Vitiligo | البَهَق | 🟢 Low (physical) |

> Each disease entry in the knowledge base includes: description, symptoms, causes, severity level, and when to see a doctor — all in both languages.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│              Streamlit Chat Interface                   │
│   (UI, session state, streaming display, PDF export)   │
└───────────────────────┬─────────────────────────────────┘
                        │ calls
                        ▼
┌─────────────────────────────────────────────────────────┐
│                      pipeline.py                        │
│                    RAG Orchestrator                     │
│                                                         │
│  1. Detect language (langdetect)                        │
│  2. Retrieve relevant chunks (retriever.py)             │
│  3. Deduplicate: 1 chunk/disease, keep top 3            │
│  4. Build bilingual system prompt with context          │
│  5. Call Groq API → LLaMA 3.3 70B (streaming)          │
│  6. Log conversation to logs.jsonl                      │
└───────────────────────┬─────────────────────────────────┘
                        │ retrieves
                        ▼
┌─────────────────────────────────────────────────────────┐
│                     retriever.py                        │
│               Semantic Vector Search                    │
│                                                         │
│  · Embedding model : BAAI/bge-m3 (multilingual)        │
│  · Index           : FAISS IndexFlatIP (cosine sim.)   │
│  · Threshold       : 0.40   ·   Top-K : 5              │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
             ▼                        ▼
    faiss_index.bin              chunks.pkl
    (vector index)           (chunk metadata)
             │                        │
             └────────────┬───────────┘
                          │ built by
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      embedder.py                        │
│              One-Time Index Builder                     │
│                                                         │
│  knowledge_base.json → chunks → embeddings → FAISS     │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
chatbot2/
│
├── app.py                 # Streamlit UI — chat interface, sidebar, PDF export
├── pipeline.py            # Core RAG logic: language detection, prompt building, LLM calls, logging
├── retriever.py           # FAISS semantic search — embeds query, returns top-k chunks with scores
├── embedder.py            # One-time script — builds FAISS index from knowledge_base.json
│
├── knowledge_base.json    # Hand-crafted medical data for all 8 diseases (EN + AR)
├── faiss_index.bin        # Pre-built FAISS vector index (output of embedder.py)
├── chunks.pkl             # Serialized chunk metadata (output of embedder.py)
│
├── logs.jsonl             # Auto-generated conversation logs (JSONL format)
├── requirements.txt       # All Python dependencies with pinned versions
└── .gitignore
```

---

## ⚙️ How It Works

### Step 1 — Knowledge Base
`knowledge_base.json` stores structured medical data for each disease with the following fields, in both English and Arabic:

- **description** — what the disease is
- **symptoms** — list of common symptoms
- **causes** — list of known causes
- **severity** — severity level and explanation
- **when_to_see_doctor** — guidance on urgency

### Step 2 — Indexing *(one-time setup)*
`embedder.py` reads the knowledge base and splits it into **text chunks** (one chunk per section per disease — e.g., *"Eczema / symptoms"*). Each chunk is embedded using `BAAI/bge-m3`, a powerful multilingual embedding model, and stored in a FAISS flat inner-product index (cosine similarity on normalized vectors).

### Step 3 — Retrieval *(every query)*
`retriever.py` encodes the user's query using the same embedding model, runs a cosine similarity search over the FAISS index, and returns the top-5 chunks that score above a threshold of `0.40`.

### Step 4 — Generation
`pipeline.py` deduplicates the results (maximum one chunk per disease, top 3 kept), builds a detailed system prompt that injects the retrieved context, and calls **LLaMA 3.3 70B** via the Groq API. Responses are streamed token by token to the UI.

### Step 5 — Display
`app.py` renders the streamed response in real time, then appends a **confidence badge** showing how well the knowledge base matched the query.

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com/)
- ~600 MB of disk space (for the embedding model, downloaded automatically on first run)

### 1. Clone the repository

```bash
git clone https://github.com/fodaa711/Skin-disease-diagnosis-.git
cd Skin-disease-diagnosis-/chatbot2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file inside the `chatbot2/` folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never commit this file to Git. It is already listed in `.gitignore`.

### 4. Build the vector index

> **Skip this step** if `faiss_index.bin` and `chunks.pkl` already exist in the folder.

```bash
python embedder.py
```

This downloads the `BAAI/bge-m3` embedding model (~570 MB on first run) and builds the FAISS index from the knowledge base. Output files `faiss_index.bin` and `chunks.pkl` are saved automatically.

---

## ▶️ Running the Chatbot

```bash
streamlit run app.py
```

Open your browser and go to **http://localhost:8501**

---

## ✨ Features

### 🌐 Bilingual Support
The chatbot automatically detects whether the user is writing in **English** or **Arabic** — including Egyptian colloquial dialect (e.g., *"ازيك"*, *"عامل ايه"*) — and responds in the same language with no configuration needed.

### ⚡ Streaming Responses
Responses are streamed token by token in real time, giving the user immediate feedback rather than waiting for the full answer to generate.

### 🏷️ Confidence Badges
After each medical response, a badge reflects how closely the answer matched the knowledge base — giving the user transparency about the reliability of the information.

### 📄 PDF Chat Export
Users can export their entire conversation as a downloadable PDF file directly from the sidebar panel.

### 🧠 Conversation Memory
The chatbot maintains context across up to 10 turns (20 messages) within a session, enabling natural follow-up questions such as *"is it contagious?"* after a disease has already been discussed.

### 📚 Source Attribution
When a response is grounded in the knowledge base, the confidence badge shows the exact disease and section it was sourced from — for example: `📚 Skin Cancer — Symptoms (82%)`.

---

## 📊 Confidence Scoring

| Badge | Score Range | Meaning |
|-------|-------------|---------|
| 🟢 High confidence | ≥ 75% | Strong match found in the knowledge base |
| 🟡 Medium confidence | 55 – 74% | Partial match found |
| 🔴 Low confidence | < 55% | Weak match; answer may be limited |
| *(no badge)* | fallback | No relevant match found |

> Confidence badges are suppressed for greetings and refusal responses, as these do not involve knowledge retrieval.

---

## 🛡️ Chatbot Boundaries

DermaCam is designed with clear, enforced guardrails to ensure safe and responsible usage:

| Situation | Chatbot Behavior |
|-----------|-----------------|
| Asked for treatment advice or medications | Declines and directs the user to a licensed dermatologist |
| Asked about a disease outside its knowledge base | Declines and lists the 8 supported conditions |
| Asked about an unrelated topic (celebrities, weather, sports) | Stays in scope and refuses politely |
| Asked to name a specific doctor or hospital | Gives general guidance only, never names specific providers |
| Receives rude or offensive input | Responds calmly and redirects to the topic |
| User expresses emotional distress about their condition | Responds with empathy and encourages professional support |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 3.0.1 | Multilingual embedding model (BAAI/bge-m3) |
| `faiss-cpu` | 1.8.0 | Fast vector similarity search |
| `groq` | 1.1.1 | Groq API client for LLaMA 3.3 70B |
| `langdetect` | 1.0.9 | Automatic language detection |
| `streamlit` | 1.37.0 | Web-based chat interface |
| `python-dotenv` | 1.0.1 | Load secrets from `.env` file |
| `numpy` | 1.26.4 | Vector mathematics |
| `fpdf2` | latest | PDF generation for chat export |

---

## 📝 Conversation Logging

All conversations are automatically appended to `logs.jsonl` in the following format:

```json
{
  "timestamp": "2026-03-22T10:29:15.074388",
  "language":  "en",
  "query":     "i have skin cancer and i am very worried about it",
  "answer":    "I'm so sorry to hear that...",
  "score":     0.6612
}
```

These logs can be used to monitor usage patterns, audit chatbot behavior, and improve the knowledge base over time.

---

## ⚠️ Disclaimer

> DermaCam is an **educational awareness tool** developed as a graduation project.
> It does **not** provide medical diagnoses, clinical assessments, or treatment recommendations.
> All information presented is for general awareness purposes only.
> **Always consult a licensed dermatologist** for any skin-related medical concern.

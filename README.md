# 🧠 AI Data Analysis Agent (RAG + LangGraph)

A portfolio-grade AI application that transforms raw tabular data into **structured insights, grounded answers, and executive summaries** using a combination of **EDA + RAG + Agent workflow orchestration**.

---

## 🚀 Why this project stands out

Unlike typical LLM demos, this project:

- ✅ Uses **deterministic data profiling (Pandas)** instead of raw text input  
- ✅ Implements a **LangGraph-based agent workflow**, not a single prompt chain  
- ✅ Applies **RAG (Retrieval-Augmented Generation)** to handle large datasets  
- ✅ Enforces **grounded reasoning** (no unsupported claims)  
- ✅ Bridges **data analysis + LLM reasoning + product experience**

👉 This is not a chatbot — it is a **data analysis copilot**

---

## 🎯 Problem it solves

Analyzing CSV datasets typically requires:
- SQL / Pandas knowledge  
- manual exploration  
- interpreting charts  

This project enables:

👉 Upload data → Ask questions → Get structured insights

---

## 🏗️ System Architecture

### Core Components

- **FastAPI** → Backend API & orchestration  
- **Pandas** → Deterministic data profiling (EDA)  
- **LangGraph** → Agent workflow coordination  
- **OpenAI API** → Reasoning + summarization  
- **Streamlit** → Demo UI  

---

### 🔁 Agent Workflow (LangGraph)
START
↓
Load Dataset
↓
Build Profile (EDA)
↓
Generate Semantic Context
↓
Retrieve Relevant Context (RAG)
↓
Route Task
├── Question Answering
└── Executive Summary
↓
END

---

## 🧩 Key Technical Design

### 1️⃣ Structured Data → Semantic Knowledge (RAG)

Instead of feeding raw tables into the LLM:

- Convert:
  - numeric summaries → text insights  
  - categorical distributions → semantic chunks  
  - missing values → data quality signals  

- Store as:
  - embeddings  
  - vectorized knowledge chunks  

👉 Enables scalable reasoning on large datasets

---

### 2️⃣ Retrieval-Augmented Generation (RAG)

At query time:

- Embed user question  
- Retrieve **top-k relevant data chunks**  
- Construct **focused context window**  

👉 Solves:
- context window limitation  
- noisy full-table input  
- long-column reasoning issues  

---

### 3️⃣ Grounded Answer Generation

The model is explicitly constrained:

- Only answer from:
  - schema  
  - statistics  
  - missing values  
  - retrieved context  

- Must return:

```json
{
  "answer": "...",
  "supporting_points": [...],
  "caveats": [...]
}
👉 Prevents hallucination and improves trustworthiness

⸻

4️⃣ LangGraph Workflow (Agent Design)

Instead of a single LLM call:
	•	Multi-step workflow:
	•	data loading
	•	profiling
	•	context construction
	•	retrieval
	•	reasoning
	•	Task routing:
	•	Q&A
	•	executive summary

👉 More controllable, explainable, and extensible

📊 Features
	•	📂 Upload CSV datasets
	•	🔍 Automatic schema & data profiling
	•	📉 Missing value analysis
	•	📊 Numeric & categorical summaries
	•	📈 Auto-generated charts
	•	💬 Natural language Q&A
	•	🧠 RAG-based reasoning
	•	📄 Executive summary for stakeholders

🧪 RAG vs Full Context (Design Insight)

This project explicitly compares:
Approach
Problem
Full context input
Token limit, noisy signals
RAG retrieval
Focused, scalable, explainable

👉 RAG improves:
	•	answer relevance
	•	consistency
	•	interpretability

🛠️ Tech Stack

Backend
	•	Python
	•	FastAPI
	•	LangGraph
	•	Pandas

LLM & RAG
	•	OpenAI API
	•	Embeddings
	•	Vector Store
	•	Retrieval pipeline

Frontend
	•	Streamlit

Data
	•	CSV-based tabular datasets

⚡ How to run
# install dependencies
pip install -r requirements.txt

# run backend
uvicorn backend.app.main:app --reload

# run frontend
streamlit run frontend/app.py

🎯 Example Use Case
Using the Diabetes dataset:
	•	Upload dataset
	•	Auto-generate statistics
	•	Ask:
	•	“Which features are most related to diabetes?”
	•	“Are there missing values?”
	•	Get:
	•	structured answers
	•	supporting evidence
	•	caveats

📌 Future Improvements
	•	Adaptive chunking strategies
	•	Dynamic grounding evidence selection
	•	Evaluation pipeline (RAG vs baseline)
	•	Multi-dataset support
	•	Caching & performance optimization

👨‍💻 Author

Li Zheng
MSc Applied Artificial Intelligence @ University of Warwick
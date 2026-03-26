# 🧠 AI Data Analysis Agent (RAG + LangGraph)

A portfolio-grade AI application that transforms raw tabular data into **structured insights, grounded answers, and executive summaries** using a combination of **EDA + RAG + agent workflow orchestration**.

---

## 🚀 Why this project stands out

Unlike typical LLM demos, this project:

- Uses **deterministic data profiling (Pandas)** instead of raw text input
- Implements a **LangGraph-based agent workflow**, not a single prompt chain
- Applies **RAG (Retrieval-Augmented Generation)** to handle large datasets
- Enforces **grounded reasoning** to reduce hallucination
- Bridges **data analysis, LLM reasoning, and product experience**

This is not a chatbot — it is a **data analysis copilot**.

---

## 🎯 Problem it solves

Analyzing CSV datasets usually requires:

- SQL or Pandas knowledge
- manual exploration
- chart interpretation

This project enables:

**Upload data → Ask questions → Get structured insights**

---

## 🏗️ System Architecture

### Core Components

- FastAPI — backend API and orchestration  
- Pandas — deterministic data profiling (EDA)  
- LangGraph — agent workflow coordination  
- OpenAI API — reasoning and summarization  
- Streamlit — demo UI  

---

### 🔁 Agent Workflow

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
→ Question Answering  
→ Executive Summary  
↓  
END  

---

## 🧩 Key Technical Design

### 1. Structured Data → Semantic Knowledge

Instead of feeding raw tables into the LLM:

- numeric summaries → text insights  
- categorical distributions → semantic chunks  
- missing values → data quality signals  

Then:
- convert to embeddings  
- store in vector database  

This enables scalable reasoning on large datasets.

---

### 2. Retrieval-Augmented Generation (RAG)

At query time:

- embed user question  
- retrieve top-k relevant chunks  
- construct focused context  

This solves:

- context window limits  
- noisy full-table input  
- long-column reasoning issues  

---

### 3. Grounded Answer Generation

The model is constrained to answer only from:

- schema  
- statistics  
- missing values  
- retrieved context  

Output format:

- answer  
- supporting_points  
- caveats  

This reduces hallucination and improves trust.

---

### 4. LangGraph Workflow

Instead of a single LLM call:

- multi-step workflow:
  - data loading  
  - profiling  
  - context construction  
  - retrieval  
  - reasoning  

- supports task routing:
  - Q&A  
  - executive summary  

This improves control and explainability.

---

## 📊 Features

- Upload CSV datasets  
- Automatic schema & profiling  
- Missing value analysis  
- Numeric & categorical summaries  
- Auto-generated charts  
- Natural language Q&A  
- RAG-based reasoning  
- Executive summaries  

---

## 🧪 RAG vs Full Context

| Approach | Limitation |
|----------|-----------|
| Full context | Token limit, noisy |
| RAG | Requires chunking design |

RAG improves:

- relevance  
- consistency  
- interpretability  

---

## 🛠️ Tech Stack

### Backend
- Python  
- FastAPI  
- LangGraph  
- Pandas  

### LLM & RAG
- OpenAI API  
- Embeddings  
- Vector Store  

### Frontend
- Streamlit  

---

## ⚡ How to Run

Install dependencies:

pip install -r requirements.txt

Run backend:

uvicorn backend.app.main:app --reload

Run frontend:

streamlit run frontend/app.py

---

## 🎯 Example Use Case

Using a healthcare dataset:

- upload data  
- auto-generate statistics  
- ask:
  - "Which features are important?"  
  - "Are there missing values?"  

System returns:

- structured answers  
- supporting evidence  
- caveats  

---

## 🔍 What Interviewers Can Explore

- Chunking strategy for structured data  
- Why RAG instead of full context  
- How hallucination is reduced  
- LangGraph vs simple chain  
- Evaluation of answer quality  

---

## 📌 Future Improvements

- adaptive chunking  
- dynamic grounding  
- evaluation pipeline  
- multi-dataset support  
- caching  

---

## 👨‍💻 Author

Li Zheng  
MSc Applied Artificial Intelligence, University of Warwick
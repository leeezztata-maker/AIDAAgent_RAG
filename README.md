# AI Data Analysis Agent

`AI Data Analysis Agent` is a portfolio-grade MVP for AI Engineer, Agent Engineer, and AI Product roles. It combines deterministic dataset profiling with LLM reasoning, so the product feels useful in a real demo instead of behaving like a generic chatbot.

## Why this project is strong for recruiters

- Shows full-stack AI product thinking: FastAPI backend, Streamlit frontend, and OpenAI integration.
- Uses grounded analysis: every answer is based on computed dataset statistics and schema metadata.
- Demonstrates agent workflow design with LangGraph rather than a single prompt wrapper.
- Solves a practical business problem: upload a CSV and get schema inspection, data quality checks, charts, Q&A, and executive summaries.

## MVP scope

- Upload CSV files
- Inspect schema and inferred data types
- Report missing values
- Generate numeric and categorical summaries
- Auto-build charts for key columns
- Answer natural-language questions about the dataset
- Generate an executive summary for non-technical users

## Architecture

- `FastAPI` serves the analysis API and orchestrates backend workflows.
- `Pandas` handles deterministic profiling and aggregation.
- `LangGraph` coordinates a grounded question-answering workflow.
- `OpenAI` turns structured analysis context into clear answers and executive summaries.
- `Streamlit` will provide the demo-friendly UI.

## Project structure

```text
ai-data-analysis-agent/
├── backend/
│   └── app/
│       ├── api/
│       ├── core/
│       ├── models/
│       ├── services/
│       └── main.py
├── data/
│   ├── exports/
│   └── uploads/
├── frontend/
│   └── app.py
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn app.main:app --app-dir backend --reload
```

Backend docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Environment variables

- `OPENAI_API_KEY`: required for Q&A and executive summary generation
- `OPENAI_MODEL`: optional, defaults to `gpt-4.1-mini`
- `DATA_DIR`: optional, defaults to `./data`
- `APP_ENV`: optional, defaults to `development`

## Next step

The backend is implemented in this iteration. The next iteration should add the Streamlit frontend, polish the user experience, and add tests.

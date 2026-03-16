# Medical AI Research Assistant — RAG System

An agentic RAG (Retrieval-Augmented Generation) system that answers questions about medical AI research papers. Built with hybrid search, reranking, HyDE, self-grounding checks, and web search fallback. Served via a FastAPI endpoint and evaluated with RAGAS.

---

## RAGAS Evaluation Scores

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.98 |
| Context Recall | 0.80 |
| Context Precision | 0.87 |

---

## Features

- **Hybrid Search** — combines ChromaDB semantic search with BM25 keyword search for better retrieval
- **CrossEncoder Reranking** — reranks retrieved chunks by relevance before passing to the LLM
- **HyDE (Hypothetical Document Embeddings)** — generates a fake ideal answer first, then searches with it for better retrieval (only triggered when retrieval confidence is poor)
- **Self-RAG Grounding Check** — verifies the answer is actually grounded in the retrieved context, retries with a rewritten query if not (max 2 retries)
- **Agentic Web Search** — falls back to Tavily web search when local documents have no relevant information
- **Domain Filter** — rejects questions unrelated to medical AI before processing
- **Sliding Window Memory** — keeps last 6 messages for multi-turn conversations
- **Source Citations** — every answer includes which research paper it came from
- **Memory & Speed Profiling** — tracks RAM usage and response time at each stage

---

## Tech Stack

| Component | Tool |
|---|---|
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Main LLM | llama-3.3-70b-versatile via Groq |
| Fast LLM (filter/HyDE) | llama-3.1-8b-instant via Groq |
| Keyword Search | BM25 (rank-bm25) |
| Web Search | Tavily |
| API Framework | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Evaluation | RAGAS |

---

## Project Structure

```
├── curves_pdf.py       # Main RAG system with chat loop
├── api.py              # FastAPI endpoint
├── eval.py             # RAGAS evaluation
├── research_papers/    # Input PDF files
├── research_vault/     # ChromaDB vector database
├── .env.example        # Environment variable template
└── .gitignore
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Adr885/medical-ai-rag.git
cd medical-ai-rag
```

**2. Install dependencies**
```bash
pip install chromadb openai tavily-python sentence-transformers rank-bm25 pymupdf langchain langchain-text-splitters fastapi uvicorn python-dotenv ragas
```

**3. Set up environment variables**

Copy `.env.example` to `.env` and add your keys:
```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Get your keys from:
- Groq: https://console.groq.com
- Tavily: https://app.tavily.com

**4. Add your PDF research papers** to the `research_papers/` folder

---

## Running

**Run the main chat system:**
```bash
python curves_pdf.py
```

**Run the FastAPI endpoint:**
```bash
uvicorn api:app --reload
```
Then send POST requests to `http://localhost:8000/ask` with body:
```json
{"text": "What is image registration?"}
```

**Run RAGAS evaluation:**
```bash
python eval.py
```

---

## How It Works

1. PDFs are parsed and split into 500-character chunks with 200-character overlap
2. Chunks are stored in ChromaDB with semantic embeddings
3. A BM25 keyword index is also built over all chunks
4. When a question arrives, both ChromaDB and BM25 retrieve candidates
5. If retrieval confidence is poor, HyDE generates a hypothetical answer and searches again
6. All candidates are merged and reranked by CrossEncoder
7. Top chunks are passed to llama-3.3-70b as context
8. The answer is grounded-checked — if it fails, the query is rewritten and retried
9. If no local context exists, Tavily web search is triggered automatically

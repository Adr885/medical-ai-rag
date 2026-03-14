from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import fitz
import glob
import os
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# --- SETUP ---
app = FastAPI()

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
db_client = chromadb.PersistentClient(path="./research_vault")
research_drawer = db_client.get_or_create_collection(name="research", embedding_function=ef)
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

all_documents = []
folder_name = r"C:\Courses\Datascience\Numpy\rag\research_papers"
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
files = glob.glob(f"{folder_name}/*.pdf")
for path in files:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    chunks = splitter.split_text(text.strip())
    all_documents.extend(chunks)

tokenized_docs = [doc.lower().split() for doc in all_documents]
bm25 = BM25Okapi(tokenized_docs)
print("✅ API Ready v2")

# --- HELPER FUNCTIONS ---
def bm25_search(query, top_n=4):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [all_documents[i] for i in top_indexes if scores[i] > 0]

def rerank_chunks(query, raw_chunks):
    if not raw_chunks:
        return ""
    pairs = [[query, chunk] for chunk in raw_chunks]
    scores = rerank_model.predict(pairs)
    scored_results = sorted(zip(scores, raw_chunks), key=lambda x: x[0], reverse=True)
    top_hits = [chunk for score, chunk in scored_results if score > 0.1]
    return "\n\n".join(top_hits[:3])

# --- ENDPOINT ---
class Question(BaseModel):
    text: str

@app.post("/ask")
def ask_question(question: Question):
    user_msg = question.text

    r_results = research_drawer.query(query_texts=[user_msg], n_results=4)

    scored_chunks = []
    for i in range(len(r_results['documents'][0])):
        if r_results['distances'][0][i] < 0.8:
            scored_chunks.append([
                r_results['documents'][0][i],
                r_results['metadatas'][0][i]['source']
            ])

    texts = [item[0] for item in scored_chunks]
    bm25_results = bm25_search(user_msg)
    combined = list(set(texts + bm25_results))
    context = rerank_chunks(user_msg, combined)

    if not context:
        return {"answer": "No relevant information found.", "sources": []}

    sources = list(set([item[1] for item in scored_chunks]))

    return {
        "answer": f"Based on research papers: {context[:500]}",
        "sources": sources
    }

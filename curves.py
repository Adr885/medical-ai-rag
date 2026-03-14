import tracemalloc
tracemalloc.start()
import json

import fitz
import os
import chromadb
import random
from chromadb.utils import embedding_functions
from openai import OpenAI
import glob
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import tracemalloc
import time
from dotenv import load_dotenv
load_dotenv()

# 1. SETUP

# --- STEP 2: SETUP THE BRAIN & WAREHOUSE ---
db_client = chromadb.PersistentClient(path="./research_vault")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Max characters per piece
    chunk_overlap=200   # Repeat 50 characters so context isn't lost
)

# Create two separate "Drawers" (Collections)
research_drawer = db_client.get_or_create_collection(name="research", embedding_function=ef)
current, peak = tracemalloc.get_traced_memory()
print(f"🧠 [CHECKPOINT 1] Model & DB Load: {current / 1024 / 1024:.2f} MB")

print("✅ Step 2: Database and Brain are ready.")





# processing files for keyword search
all_documents = []
all_ids = []
folder_name = r"C:\Courses\Datascience\Numpy\rag\research_papers"
files = glob.glob(f"{folder_name}/*.pdf")
for path in files:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.strip()
    chunks = splitter.split_text(text)
    if not chunks and text:
        chunks = [text]
    for i, chunk in enumerate(chunks):
        all_documents.append(chunk)
        all_ids.append(f"{os.path.basename(path)}_{i}")

# BM25 needs words, not sentences
tokenized_docs = [doc.lower().split() for doc in all_documents]
bm25 = BM25Okapi(tokenized_docs)
print("✅ BM25 Keyword Index built.")

# --- STEP 3: THE INHALER (Moving files to Database) ---
folder_name = r"C:\Courses\Datascience\Numpy\rag\research_papers"
files = glob.glob(f"{folder_name}/*.pdf")
for path in files:
    file_name = os.path.basename(path)
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = text.strip()

    chunks = splitter.split_text(text)
    
    # 1. THE SAFETY FALLBACK: Must be indented inside the 'files' loop
    if not chunks and text:
        chunks = [text]
    
    count = 0 
    # 2. THE SINGLE LOOP: Don't write 'for piece in chunks' twice
    for piece in chunks:
      unique_id = f"{file_name}_{count}"
      research_drawer.upsert(
        documents=[piece],
        ids=[unique_id],
        metadatas=[{"source": file_name}]
    )
      count += 1

    print(f"✅ Inhaled {file_name} into {count} vectors.")
    
current, peak = tracemalloc.get_traced_memory()
print(f"📂 [CHECKPOINT 2] Post-Ingestion RAM: {current / 1024 / 1024:.2f} MB")


## reranker fuctuons
before_rerank, _ = tracemalloc.get_traced_memory()
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
after_rerank, _ = tracemalloc.get_traced_memory()
rerank_cost = (after_rerank - before_rerank) / 1024 / 1024
print(f"🤖 [CHECKPOINT 1.5] CrossEncoder cost exactly: {rerank_cost:.2f} MB")
print(f"   Total RAM now: {after_rerank / 1024 / 1024:.2f} MB")


## function of bm25
def bm25_search(query, top_n=4):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # Get indexes of top scoring documents
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = []
    for i in top_indexes:
        if scores[i] > 0:  # Only include if there's any keyword match
            results.append(all_documents[i])
    return results



def rerank_chunks(query, raw_chunks):
    if not raw_chunks:
        return ""

    # Step 1: Create pairs of [Query, Chunk] for the judge to evaluate
    pairs = [[query, chunk] for chunk in raw_chunks]
    
    # Step 2: The model predicts a relevance score (weight) for each pair
    scores = rerank_model.predict(pairs)
    
    # Step 3: Combine chunks with their scores and sort by highest weight
    # We use a threshold of 0.3 to filter out "hallucination-bait" (useless data)
    scored_results = sorted(zip(scores, raw_chunks), key=lambda x: x[0], reverse=True)
    top_hits = [chunk for score, chunk in scored_results if score > 0.1]
    
    # Step 4: Return only the top 3 high-quality chunks
    return "\n\n".join(top_hits[:3])





## checking if the ans come from the documents or not
def is_grounded(answer, chunks):
    if not chunks:
        return False
    
    context = "\n".join(chunks[:3])
    
    response = ai_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content":"check if answer is related to context .do not be too strict just check if it is somehow relevant to context.answer only yes or no"
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nAnswer: {answer}\n\nIs every statement in the answer supported by the context?"
            }
        ]
    )
    
    result = response.choices[0].message.content.strip().lower()
    return "yes" in result

## rewriting the query if needed

def rewrite_query(original_question, failed_answer):
    response = ai_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "The current search query failed to find good information. Rewrite it to be more specific and technical to find better results. Return only the rewritten query, nothing else."
            },
            {
                "role": "user",
                "content": f"Original question: {original_question}\nFailed answer: {failed_answer}\nRewrite the search query:"
            }
        ]
    )
    return response.choices[0].message.content.strip()
    

## API calling of openai and travity



ai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query):
    print(f"\n[SYSTEM] Agent is searching the web for: '{query}'...")
    
    # This fetches the top context snippets from across the web
    # 'search_depth="advanced"' ensures high-quality results
    result = tavily.get_search_context(query=query,  search_depth="basic", max_tokens=500)
    
    return result

# Function to shrink the history


## description of the function
tools = [
    
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description":  "Search the web ONLY when the context explicitly says 'No specific records found.' Never use this for questions about hospital policies, patients, or anything that could be in local records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    } # <-- Fixed the closing brackets here
]

## check if ans relevant to question or not
def is_relevant(question):
    response = ai_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a filter for a medical AI research assistant. Reply YES for any question about: medical imaging, AI models, machine learning, deep learning, transformers, neural networks, segmentation, image registration, research papers, datasets, model performance, technical concepts. Reply NO only for completely unrelated topics like math, sports, geography, jokes. Reply ONLY with yes or no."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    result = response.choices[0].message.content.strip().lower()
    return "yes" in result


# 3. CHAT LOOP
memory = [{"role": "system", "content": "You are a research assistant. Answer ONLY from the context provided. If the context contains relevant information, use it directly and do NOT call web_search. Only call web_search if the context says 'No specific records found."}]



while True:
    user_msg = input("\nYou: ")
    if not user_msg.strip():
        print("AI: Please ask a question.")
        continue
    if user_msg.lower() == "exit": 
        break

    if not is_relevant(user_msg):
        print("AI: I'm a medical research assistant. I can only answer questions related to medical AI and research papers.")
        continue

    total_start = time.time()

    # STEP 1: DATABASE SEARCH
    
    db_start = time.time()
    try:
       r_results = research_drawer.query(query_texts=[user_msg], n_results=4)
    except Exception as e:
       print(f"AI: Search failed. Please try again.")
       continue
    db_end = time.time()
    print(f"⏱️ ChromaDB retrieval took: {(db_end - db_start) * 1000:.2f} ms")

    THRESHOLD = 0.8
    scored_chunks = []

    # 2. THE CHECK (The Bouncer)
    for i in range(len(r_results['documents'][0])):
     dist = r_results['distances'][0][i]
     if dist < THRESHOLD:
         scored_chunks.append([
         r_results['documents'][0][i],
         r_results['ids'][0][i],
         r_results['metadatas'][0][i]['source']
])
    top_chunks = [item[0] for item in scored_chunks] 


    texts_for_rerank = [item[0] for item in scored_chunks]
    bm25_results = bm25_search(user_msg, top_n=4)
    print(f"🔑 BM25 found {len(bm25_results)} keyword matches")

# Combine both lists, remove duplicates
    combined = list(set(texts_for_rerank + bm25_results))
    texts_for_rerank = combined


    start = time.time()
    reranked_context = rerank_chunks(user_msg, texts_for_rerank)
    end = time.time()
    print(f"⏱️ Reranking took: {(end - start) * 1000:.2f} ms")

    current, peak = tracemalloc.get_traced_memory()
    print(f"🔍 [CHECKPOINT 2.5] After reranking query: {current / 1024 / 1024:.2f} MB")

    # 2. Use that directly as your context
    context = reranked_context if reranked_context else "No specific records found."

    # 3. CHAT (The AI will now see the context and decide if it needs Web Search)
    sources = list(set([item[2] for item in scored_chunks]))
    prompt_with_context = f"Context: {context}\n\nSources: {sources}\n\nUser Question: {user_msg}\n\nNote: mention which source PDF your answer came from."
    
   
    # Keep system prompt + last 6 messages only
    if len(memory) > 7:
     memory = [memory[0]] + memory[-6:]

    memory.append({"role": "user", "content": prompt_with_context})
    try:
        response = ai_client.chat.completions.create(
         model="llama-3.3-70b-versatile",
         messages=memory,
         tools=tools,         
         tool_choice="auto",
         temperature=0
    )
    except Exception as e:
      print(f"AI: Could not generate response. Please try again.")
      continue
    ai_msg = response.choices[0].message

    # --- THE SWITCHBOARD ---
    if ai_msg.tool_calls:
        memory.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
        

            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = web_search(args['query']) 
                content_result = str(result)
                print(f"\n[SYSTEM] Web Search completed.")

            memory.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": content_result
            })

        # Get final response after tools
        final_response = ai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=memory
        )
        answer = final_response.choices[0].message.content

    else:
        # If no tools were needed
        answer = ai_msg.content
        max_retries = 2
        retry_count = 0

    while not is_grounded(answer, top_chunks) and retry_count < max_retries:
        retry_count += 1
        print(f"[Self-RAG] Answer not grounded, retry {retry_count}/{max_retries}...")
    
        better_query = rewrite_query(user_msg, answer)
    
        new_results = research_drawer.query(query_texts=[better_query], n_results=4)
        new_chunks = [new_results['documents'][0][i] for i in range(len(new_results['documents'][0])) if new_results['distances'][0][i] < 0.9]
        new_context = rerank_chunks(user_msg, new_chunks)
    
        retry_response = ai_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context."},
            {"role": "user", "content": f"Context: {new_context}\n\nQuestion: {user_msg}"}
        ]
    )
        answer = retry_response.choices[0].message.content

    if retry_count == max_retries:
        print("[Self-RAG] Max retries reached, returning best answer")
    else:
        print("[Self-RAG] Answer grounded ✅")

    # This runs for BOTH cases (if tool or no tool)
    print(f"\nAI: {answer}")
    memory.append({"role": "assistant", "content": answer})
    total_end = time.time()
    print(f"⏱️ Total response time: {(total_end - total_start):.2f} seconds")


    current, peak = tracemalloc.get_traced_memory()
    print(f"💬 [CHECKPOINT 3] Active Chat RAM: {current / 1024 / 1024:.2f} MB")
    print(f"🔝 Peak Memory reached so far: {peak / 1024 / 1024:.2f} MB")

    # --- THE SUMMARY TRIGGER (Inside the while loop) ---
  
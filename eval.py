import os
import chromadb
import fitz
import glob
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


# --- SETUP ---
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
wrapped_embeddings = LangchainEmbeddingsWrapper(huggingface_embeddings)
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

ai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
print("✅ Setup complete")


def bm25_search(query, top_n=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [all_documents[i] for i in top_indexes if scores[i] > 0]


## so function to check query expension needed or not
def needs_expansion(question):
    # Short simple questions don't need expansion
    word_count = len(question.split())
    if word_count <= 4:
        return False
    
    # Simple definition questions don't need expansion
    simple_starters = ["what is", "define", "who is"]
    for starter in simple_starters:
        if question.lower().startswith(starter) and word_count <= 5:
            return False
    
    return True

##for actual query expension
def expand_query(question):
    response = ai_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 alternative phrasings of the given question. Return only the 3 questions, one per line, no numbering, no extra text."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    alternatives = response.choices[0].message.content.strip().split("\n")
    all_queries = [question] + alternatives
    return all_queries


def rerank_chunks(query, raw_chunks):
    if not raw_chunks:
        return [], ""
    pairs = [[query, chunk] for chunk in raw_chunks]
    scores = rerank_model.predict(pairs)
    scored_results = sorted(zip(scores, raw_chunks), key=lambda x: x[0], reverse=True)
    top_hits = [chunk for score, chunk in scored_results if score > 0.3]
    return top_hits, "\n\n".join(top_hits[:3])

test_questions = [
    {
        "question": "What is the main limitation of ConvNet in medical image analysis?",
        "ground_truth": "ConvNet has a limited effective receptive field due to the locality of convolution operations, making it unable to model long-range spatial relations between distant voxels."
    },
    {
        "question": "What is image registration?",
        "ground_truth": "Image registration is the process of establishing spatial correspondence between moving and fixed images by comparing different parts of the moving image to the fixed image."
    },
    {
        "question": "Why is Transformer better than ConvNet for image registration?",
        "ground_truth": "Transformer has large effective receptive fields and self-attention mechanisms that capture long-range spatial information, handling large deformations better than ConvNet."
    },
    {
        "question": "What is TransMorph?",
        "ground_truth": "TransMorph is a hybrid Transformer-ConvNet framework for volumetric medical image registration that bridges ViT and V-Net."
    },
    {
        "question": "What is oracle performance in SAM?",
        "ground_truth": "Oracle performance is when the prediction closest to the true mask is always selected from SAM's three generated predictions, representing the upper bound of performance."
    },
    {
        "question": "How does SAM perform compared to other methods?",
        "ground_truth": "SAM performed better than all other methods on 24 out of 28 tasks and in oracle mode was better on 26 out of 28 tasks."
    },
]


# hyde_search
def hyde_search(question):
    # Step 1: Generate fake perfect answer
    response = ai_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Generate a detailed technical paragraph that would perfectly answer this question. Use domain specific terminology."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    fake_answer = response.choices[0].message.content
    print(f"[HyDE] Generated hypothesis: {fake_answer[:100]}...")
    
    # Step 2: Search with fake answer instead of question
    results = research_drawer.query(query_texts=[fake_answer], n_results=4)
    
    hyde_chunks = []
    for i in range(len(results['documents'][0])):
        if results['distances'][0][i] < 0.9:
            hyde_chunks.append(results['documents'][0][i])
    
    return hyde_chunks

def run_rag(question):
    if needs_expansion(question):
       queries = expand_query(question)
       all_chroma_chunks = []
       for q in queries:
         r_results = research_drawer.query(query_texts=[q], n_results=4)
         for i in range(len(r_results['documents'][0])):
           if r_results['distances'][0][i] < 0.9:
             all_chroma_chunks.append(r_results['documents'][0][i])

# Remove duplicates
       chroma_chunks = list(set(all_chroma_chunks))
       best_score = min(r_results['distances'][0])
    else:
       queries = [question]

       r_results = research_drawer.query(query_texts=[question], n_results=4)

       best_score = min(r_results['distances'][0])
       print(f"[Retrieval confidence: {best_score:.3f}]")
    
       chroma_chunks = []
       for i in range(len(r_results['documents'][0])):
         if r_results['distances'][0][i] < 0.9:
            chroma_chunks.append(r_results['documents'][0][i])

    

    bm25_results = bm25_search(question, top_n=5) 
    if best_score > 0.7:
       print("[HyDE triggered - poor retrieval]")
       hyde_chunks = hyde_search(question)
    else:
       print("[HyDE skipped - good retrieval]")
       hyde_chunks = []
    
    combined = list(set(chroma_chunks + bm25_results + hyde_chunks))
    
    top_chunks, context = rerank_chunks(question, combined)
    
    if not context:
        return "No relevant information found.", []
    
    response = ai_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer the question directly and concisely using only the provided context. Stay focused on exactly what was asked."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    
    answer = response.choices[0].message.content
    return answer, top_chunks

# --- RUN EVALUATION ---
print("Running evaluation...")

questions = []
answers = []
contexts = []
ground_truths = []

for i, test in enumerate(test_questions):
    print(f"Testing question {i+1}/{len(test_questions)}...")
    answer, retrieved_chunks = run_rag(test['question'])
    questions.append(test['question'])
    answers.append(answer)
    contexts.append(retrieved_chunks)
    ground_truths.append(test['ground_truth'])

print("✅ All questions tested, now scoring...")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


groq_llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
     api_key=os.getenv("GROQ_API_KEY")

)


wrapped_llm = LangchainLLMWrapper(groq_llm)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=wrapped_llm,
    embeddings=wrapped_embeddings
)
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📊 RAGAS EVALUATION RESULTS")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(results)
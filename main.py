import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Tuple
import markdown2
import chromadb 
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Literal 

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.environ.get("KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

llm = ChatGroq(model_name="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY, temperature=0.0)
session_histories: Dict[str, List[Tuple[str, str]]] = {}
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=DB_DIR)

@app.on_event("startup")
def startup_event():
    print("--- Application starting up: Checking for new documents to ingest... ---")
    
    existing_collections = {collection.name for collection in chroma_client.list_collections()}
    print(f"Existing collections in DB: {existing_collections}")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
        return

    for doc_file in os.listdir(DATA_DIR):
        if doc_file.endswith(".txt"):
            collection_name = os.path.splitext(doc_file)[0].lower()
            
            if collection_name not in existing_collections:
                print(f"[INGESTING]: New document found: '{doc_file}'. Creating collection '{collection_name}'...")
                
                # 1. Load the document
                doc_path = os.path.join(DATA_DIR, doc_file)
                loader = TextLoader(doc_path, encoding='utf-8')
                documents = loader.load()
                
                # 2. Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # 3. Create the collection in ChromaDB
                Chroma.from_documents(
                    client=chroma_client,
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=collection_name
                )
                print(f"✅ [SUCCESS]: Ingestion complete for '{collection_name}'.")
    print("--- Startup ingestion check complete. ---")


CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional Relationship Manager's assistant. Your name is RaMi which stands for relantionship manager and AI, Your task is to answer the user's question based *only* on the provided context.\n"
               "Respond to greetings in friendly tone. \n"
               "**CRITICAL INSTRUCTION: You MUST respond in the following language: {language_name}.**\n"
               "Use Markdown for formatting, such as bolding key terms (`**term**`), to improve readability.\n"
               "Do not use tables to present the information, use bullet points instead. \n"
               "If the context does not contain the answer, state 'The provided information does not contain the answer to this question.'\n"
               "CONTEXT:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
class ChatRequest(BaseModel):
    query: str
    session_id: str
    client_id: str
    language: Literal['en', 'ar'] = 'en'
class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/", response_class=HTMLResponse)
async def get_chat_ui(request: Request):
    """Serves the main chat UI and dynamically finds available clients."""
    client_names = []
    try:
        collections = chroma_client.list_collections()
        client_names = [
            {"id": col.name, "name": col.name.replace('_', ' ').title()} 
            for col in collections
        ]
    except Exception as e:
        print(f"Error loading client collections: {e}")
    return templates.TemplateResponse("index.html", {"request": request, "clients": client_names})

@app.post("/chat", response_model=ChatResponse)
async def handle_chat_message(request: ChatRequest):
    """Handles incoming chat messages and returns the RAG-powered response."""
    try:
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=request.client_id,
            embedding_function=embeddings,
        )
        if request.client_id == "all":
            print("--- Querying MASTER collection. Using k=8 for comprehensive search. ---")
            retriever = vectorstore.as_retriever(search_kwargs={'k': 27})
        else:
            print(f"--- Querying SINGLE client collection: {request.client_id}. Using k=4. ---")
            retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    except Exception as e:
        print(f"Error loading vector store for client '{request.client_id}': {e}")
        raise HTTPException(status_code=404, detail=f"Knowledge base for client '{request.client_id}' not found.")

    chat_history_tuples = session_histories.get(request.session_id, [])
    chat_history_messages = [
        (HumanMessage(content=q), AIMessage(content=a)) for q, a in chat_history_tuples
    ]
    chat_history_messages = [msg for pair in chat_history_messages for msg in pair]

    condense_question_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    
    def get_retrieved_docs(inputs):
        if inputs.get("chat_history"):
            standalone_question = condense_question_chain.invoke({
                "chat_history": inputs["chat_history"],
                "question": inputs["question"]
            })
            return retriever.invoke(standalone_question)
        else:
            return retriever.invoke(inputs["question"])

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    conversational_rag_chain = RunnablePassthrough.assign(context=get_retrieved_docs) | answer_chain
    language_map = {"en": "English", "ar": "Arabic"}
    language_name = language_map.get(request.language, "English") 
    print(f"--- Responding in: {language_name} ---")

    answer = conversational_rag_chain.invoke({
        "question": request.query,
        "chat_history": chat_history_messages,
        "language_name": language_name

    })
    
    session_histories.setdefault(request.session_id, []).append((request.query, answer))
    
    html_answer = markdown2.markdown(answer)
    return ChatResponse(answer=html_answer, session_id=request.session_id)

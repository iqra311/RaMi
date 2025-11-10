import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Tuple
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
import markdown2
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Load Environment Variables ---
load_dotenv()

# --- App Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---  FastAPI App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Groq LLM
llm = ChatGroq(
    model_name="openai/gpt-oss-120b", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.0 
)

session_histories: Dict[str, List[Tuple[str, str]]] = {}

# ---  Prompt for Condensing History ---
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# --- Prompt for Answering ---
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional Relationship Manager's assistant. Your name is RaMi which stands for relantionship manager and AI, Your task is to answer the user's question based *only* on the provided context.\n"
               "You can respond to user queries in english or arabic depending on the user language. \n"
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
class ChatResponse(BaseModel):
    answer: str
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def get_chat_ui(request: Request):
    print("--- GET /: Attempting to load clients from ChromaDB... ---")
    client_names = []
    try:
        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        collections = chroma_client.list_collections()
        client_names = [
            {"id": col.name, "name": col.name.replace('_', ' ').title()} 
            for col in collections
        ]
        print(f"--- SUCCESS: Found clients: {client_names} ---")
    except Exception as e:
        print(f"--- ERROR: Could not load client collections: {e} ---")
        client_names = []
    return templates.TemplateResponse("index.html", {"request": request, "clients": client_names})
MASTER_COLLECTION_NAME="All_Clients"
@app.post("/chat", response_model=ChatResponse)
async def handle_chat_message(request: ChatRequest):
    try:
        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        vectorstore = Chroma(
            client=chroma_client, collection_name=request.client_id, embedding_function=embeddings
        )
        if request.client_id == MASTER_COLLECTION_NAME:
            print("--- Querying MASTER collection. Using k=8 for comprehensive search. ---")
            retriever = vectorstore.as_retriever(search_kwargs={'k': 27})
        else:
            print(f"--- Querying SINGLE client collection: {request.client_id}. Using k=4. ---")
            retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    except Exception:
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
            print(f"Standalone Question: {standalone_question}")
            return retriever.invoke(standalone_question)
        else:
            return retriever.invoke(inputs["question"])

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    
    conversational_rag_chain = (
        RunnablePassthrough.assign(
            context=get_retrieved_docs,
        )
        | answer_chain
    )

    answer = conversational_rag_chain.invoke({
        "question": request.query,
        "chat_history": chat_history_messages
    })

    session_histories.setdefault(request.session_id, []).append((request.query, answer))
    
    html_answer = markdown2.markdown(answer)
    
    return ChatResponse(answer=html_answer, session_id=request.session_id)

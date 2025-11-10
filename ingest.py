import os
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

print("--- Script starting ---")

# --- Configuration ---
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" 

def main(doc_path, collection_name):
    print(f"[main function] Starting ingestion process for document: {doc_path}")
    print(f"[main function] Target ChromaDB collection: {collection_name}")

    # 1. Load the document
    try:
        loader = TextLoader(doc_path, encoding='utf-8')
        documents = loader.load()
        print(f"[main function] Successfully loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"[main function] Error loading document: {e}")
        return

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[main function] Split document into {len(chunks)} chunks.")

    # 3. Initialize the embedding model
    print(f"[main function] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("[main function] Embedding model loaded.")

    # 4. Create and persist the vector store
    print(f"[main function] Creating vector store in: {DB_DIR}")
    # This will create the collection, generate embeddings, and save to disk.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_DIR
    )
    print("-" * 50)
    print("✅ Ingestion complete!")
    print(f"Vector store for client '{collection_name}' is ready and saved to disk.")
    print("-" * 50)

if __name__ == "__main__":
    print("--- main__ block ---")
    
    parser = argparse.ArgumentParser(
        description="Ingest a document into a ChromaDB collection."
    )
    parser.add_argument(
        "doc_path", 
        type=str, 
        help="Path to the document file (e.g., 'data/qair_aviation_group.txt')."
    )
    parser.add_argument(
        "collection_name", 
        type=str, 
        help="Unique name for the client's collection (e.g., 'qair_aviation')."
    )
    args = parser.parse_args()
    
    print(f"--- Parsed arguments: doc_path='{args.doc_path}', collection_name='{args.collection_name}' ---")

    if not os.path.exists(args.doc_path):
        print("🛑 CRITICAL ERROR: Document path does not exist!")
        print(f"   Your provided path: '{args.doc_path}'")
        print(f"   Current working directory: '{os.getcwd()}'")
        print("   Please check for typos or incorrect path.")
    else:
        print("--- Document path exists. Calling main function... ---")
        main(args.doc_path, args.collection_name)
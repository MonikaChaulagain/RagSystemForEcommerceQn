import json
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_chunks_from_json(json_path: str) -> list[Document]:
    """Load JSON chunk data and convert it back into LangChain Document objects."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"File not found at '{json_path}'. "
            "Ensure you have run 'src/ingestion/chunking.py' first."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    documents = []
    for chunk in chunks_data:
        # Create a standard LangChain Document with content & metadata
        doc = Document(
            page_content=chunk["content"],
            metadata=chunk.get("metadata", {}),
        )
        documents.append(doc)
    return documents


def store_in_chromadb(
    documents: list[Document],
    persist_dir: str = "data/chroma_db",
    collection_name: str = "ecommerce_marketplaces",
):
    """
    Generate embeddings for documents using BGE-M3 (open-source, multilingual,
    supports dense + sparse-style retrieval quality) and save them into a
    persistent local Chroma vector database.
    """
    print(" Loading local embedding model ('BAAI/bge-m3')...")
    # BGE-M3 runs locally on CPU/GPU without needing an API key.
    # normalize_embeddings=True is important for BGE models — they're
    # trained/evaluated with cosine similarity on normalized vectors.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f" Embedding {len(documents)} chunks and saving to '{persist_dir}'...")
    # Initialize and persist Chroma vector store
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(
        f"Successfully stored embeddings in ChromaDB collection '{collection_name}'!"
    )
    return vector_db


def main():
    # Relative paths from project root
    chunks_path = os.path.join("data", "processed", "chunks.json")
    chroma_db_dir = os.path.join("data", "chroma_db")

    print(" Loading chunks from JSON file...")
    documents = load_chunks_from_json(chunks_path)
    print(f" Successfully loaded {len(documents)} documents.")

    # Ingest chunks into ChromaDB
    store_in_chromadb(
        documents=documents,
        persist_dir=chroma_db_dir,
        collection_name="ecommerce_marketplaces",
    )


if __name__ == "__main__":
    main()
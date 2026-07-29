import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_vector_store(
    persist_dir: str = "data/chroma_db",
    collection_name: str = "ecommerce_marketplaces",
):
    """Load the existing Chroma vector store using the same embedding model."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Vector database not found at '{persist_dir}'. "
            "Please run 'src/embeddings/store_embeddings.py' first."
        )
    # Note: Use the EXACT same embedding model + settings used during ingestion!
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vector_db


def test_similarity_search(vector_db, query: str, top_k: int = 3):
    """
    Perform similarity search and print content along with relevance scores.
    In ChromaDB via LangChain, lower similarity score = closer vector match
    (Chroma's default distance metric here is squared L2, not cosine —
    since BGE-M3 embeddings are normalized, smaller distance still means
    more similar vectors).
    """
    print(f"\n🔎 Query: '{query}'")
    print("=" * 60)
    # similarity_search_with_score returns tuples of (Document, score)
    results = vector_db.similarity_search_with_score(query, k=top_k)
    if not results:
        print("⚠️ No relevant chunks found.")
        return
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"📌 Rank {rank} | Similarity Score: {score:.4f}")
        if doc.metadata:
            print(f"   Metadata: {doc.metadata}")
        print("   Content Sample:")
        # Display sample content
        content_preview = doc.page_content.strip().replace("\n", " ")
        print(f"   \"{content_preview[:300]}...\"")
        print("-" * 60)


def main():
    db_dir = os.path.join("data", "chroma_db")
    print("⚡ Loading vector database...")
    vector_db = load_vector_store(persist_dir=db_dir)

    # List of queries tailored to your document context
    sample_queries = [
        "Which multi-vendor marketplace platform is best suited for enterprise level?",
        "What are the main features and capabilities of marketplace software?",
        "How do commission fees or vendor management work across platforms?",
    ]

    for query in sample_queries:
        test_similarity_search(vector_db, query=query, top_k=2)


if __name__ == "__main__":
    main()
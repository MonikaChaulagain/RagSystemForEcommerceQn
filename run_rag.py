# run_rag.py
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.extract import main as run_extract
from src.ingestion.chunking import main as run_chunking
from src.embeddings.store_embeddings import main as run_store
from src.retrieval.rerank_and_answer import load_vector_store, retrieve_and_rerank, generate_answer


def ensure_pipeline_runs(rebuild_db: bool = False):
    """Ensure all components of the ingestion and embedding pipeline are executed in order."""
    extracted_md = PROJECT_ROOT / "data" / "processed" / "extracted_output.md"
    chunks_json = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    chroma_db = PROJECT_ROOT / "data" / "chroma_db"

    # Step 1: Extract text from PDF
    if not extracted_md.exists():
        print("\n=== [Step 1] PDF Extractor: Output markdown not found. Extracting now... ===")
        run_extract()
    else:
        print("\n=== [Step 1] PDF Extractor: Extracted markdown exists. Skipping. ===")

    # Step 2: Chunk markdown
    if not chunks_json.exists():
        print("\n=== [Step 2] Chunking: Chunks JSON not found. Chunking now... ===")
        run_chunking()
    else:
        print("\n=== [Step 2] Chunking: Chunks JSON exists. Skipping. ===")

    # Step 3: Generate and store embeddings
    if not chroma_db.exists() or rebuild_db:
        if rebuild_db:
            print("\n=== [Step 3] Vector Store: Rebuild flag passed. Creating vector store... ===")
        else:
            print("\n=== [Step 3] Vector Store: Chroma DB folder not found. Creating vector store... ===")
        run_store()
    else:
        print("\n=== [Step 3] Vector Store: Chroma DB exists. Skipping embedding step. ===")


def query_pipeline(vector_db, query: str):
    """Execute the retrieval, reranking, and generation pipeline for a single query."""
    print(f"\n[Query] Query: '{query}'")
    print("=" * 70)
    
    print("[-] Retrieving and Reranking chunks...")
    try:
        reranked = retrieve_and_rerank(vector_db, query, initial_k=10, top_n=3)
    except Exception as e:
        print(f"[Error] Error during retrieval and reranking: {e}")
        return

    if not reranked:
        print("[Warning] No relevant chunks found in the document.")
        return

    print("-" * 70)
    print("[-] Retrieved Context References (Top 3):")
    for rank, (doc, score) in enumerate(reranked, start=1):
        print(f"  Rank {rank} | Rerank Score: {score:.4f}")
        print(f"    Section: {doc.metadata.get('section_path', 'Unknown Section')}")
        excerpt = doc.page_content.strip().replace('\n', ' ')
        print(f"    Content Preview: \"{excerpt[:180]}...\"")
        print("-" * 70)

    print("[-] Querying Groq for synthesized response...")
    answer = generate_answer(query, reranked)
    print("\n[Synthesized Response]:\n")
    print(answer)
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified RAG execution pipeline for eCommerce marketplaces Q&A.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the Chroma Vector Database from scratch.")
    parser.add_argument("--query", type=str, help="Run a single query and exit, instead of entering the interactive shell.")
    args = parser.parse_args()

    # Verify that .env exists, warn if not
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("WARNING: No .env file found at the project root. Groq queries will fall back to local summarization.")
        print("   Create a .env file with 'GROQ_API_KEY=your_key' to use Groq LLM services.")
    else:
        # Check if GROQ_API_KEY is defined in the environment or file
        from dotenv import load_dotenv
        load_dotenv(env_file)
        if not os.getenv("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY is not defined in your environment or .env file.")

    # 1. Run pipeline elements to guarantee data/chroma_db is populated
    ensure_pipeline_runs(rebuild_db=args.rebuild)

    # 2. Load vector database
    print("\n[-] Loading vector store into memory...")
    try:
        vector_db = load_vector_store()
    except Exception as e:
        print(f"[Error] Critical Error loading vector database: {e}")
        print("Try running again with --rebuild to generate fresh embeddings.")
        sys.exit(1)

    # 3. Query execution
    if args.query:
        query_pipeline(vector_db, args.query)
    else:
        print("\n" + "*" * 60)
        print("Welcome to MarketplaceGuide Interactive RAG Shell!")
        print("Ask any questions about the 11 eCommerce platforms.")
        print("Type 'exit' or 'quit' (or press Enter) to close the shell.")
        print("*" * 60 + "\n")
        
        while True:
            try:
                user_query = input("Ask a question: ").strip()
                if not user_query or user_query.lower() in ("exit", "quit"):
                    print("Exiting. Thank you!")
                    break
                query_pipeline(vector_db, user_query)
            except KeyboardInterrupt:
                print("\nExiting. Thank you!")
                break


if __name__ == "__main__":
    main()

# src/retrieval/rerank_and_answer.py
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def load_vector_store(
    persist_dir: str = "data/chroma_db",
    collection_name: str = "ecommerce_marketplaces",
):
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Vector database not found at '{persist_dir}'.")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

import json
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# Global cache for BM25 to avoid reloading/retokenizing on every query
_bm25_instance = None
_all_documents = []


def get_bm25_retriever(chunks_json_path: str = "data/processed/chunks.json"):
    """Initialize or load cached BM25 index over processed chunks."""
    global _bm25_instance, _all_documents
    if _bm25_instance is not None:
        return _bm25_instance, _all_documents

    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / chunks_json_path
    if not full_path.exists():
        return None, []

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        _all_documents = []
        tokenized_corpus = []

        for chunk in chunks_data:
            doc = Document(
                page_content=chunk["content"],
                metadata=chunk.get("metadata", {}),
            )
            _all_documents.append(doc)
            # Tokenize body text for BM25
            tokenized_corpus.append(chunk["content"].lower().split())

        _bm25_instance = BM25Okapi(tokenized_corpus)
        return _bm25_instance, _all_documents
    except Exception as e:
        print(f"[Warning] Failed to initialize BM25 retriever: {e}")
        return None, []


def retrieve_and_rerank(vector_db, query: str, initial_k: int = 10, top_n: int = 3):
    """
    Stage 1: Cast a wide net using Hybrid Search (Dense Vector search + BM25 Keyword search).
    Stage 2: Merge and deduplicate candidate documents.
    Stage 3: Rerank the unified candidate list with a Cross-Encoder for precision.
    """
    # 1. Retrieve semantic matches via Vector Similarity Search
    vector_candidates_with_score = vector_db.similarity_search_with_score(query, k=initial_k)
    vector_candidates = [doc for doc, _ in vector_candidates_with_score]

    # 2. Retrieve keyword matches via BM25
    bm25, all_docs = get_bm25_retriever()
    bm25_candidates = []
    if bm25 is not None:
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        # Sort and take top matches with non-zero scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:initial_k]
        bm25_candidates = [all_docs[i] for i in top_indices if scores[i] > 0]

    # 3. Merge and deduplicate candidates, keeping relative ranking order
    seen_contents = set()
    combined_candidates = []

    for doc in vector_candidates + bm25_candidates:
        # Deduplicate based on exact page content
        content_hash = doc.page_content.strip()
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            combined_candidates.append(doc)

    if not combined_candidates:
        return []

    # 4. Rerank the combined candidates list using Cross-Encoder
    try:
        from sentence_transformers import CrossEncoder

        # Use BAAI/bge-reranker-base, which is fast and fits in memory on CPU
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        pairs = [(query, doc.page_content) for doc in combined_candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(combined_candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
    except Exception as exc:
        print(f"[Warning] Reranker unavailable ({exc}). Falling back to similarity ordering.")
        # Fallback to vector ranking order or BM25 order if reranker fails
        return [(doc, 1.0) for doc in combined_candidates[:top_n]]


def build_grounded_fallback_answer(query: str, context_blocks: list[str]) -> str:
    """Create a synthesized, structured summary from retrieved context when Groq is unavailable."""
    if not context_blocks:
        return "No relevant context was retrieved from the database to answer this question."

    summary_lines = [
        "Groq LLM is currently unavailable (fallback active). Here is a synthesized summary from the retrieved document sections:",
        ""
    ]
    for block in context_blocks:
        lines = block.splitlines()
        if not lines:
            continue
        section_header = lines[0]
        content_body = "\n".join(lines[1:]).strip()
        # Clean/truncate preview
        content_preview = " ".join(content_body.split())
        if len(content_preview) > 300:
            content_preview = content_preview[:300] + "..."
        summary_lines.append(f"- {section_header}: {content_preview}")
        summary_lines.append("")

    return "\n".join(summary_lines).strip()


def generate_answer(query: str, reranked_chunks):
    """Create a grounded answer from the retrieved context when possible."""
    context_blocks = []
    for doc, score in reranked_chunks:
        section = doc.metadata.get("section_path", "Unknown section")
        context_blocks.append(f"[Section: {section}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_blocks)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "No GROQ_API_KEY found in the environment. "
            f"Fallback answer:\n{build_grounded_fallback_answer(query, context_blocks)}"
        )

    try:
        from langchain_groq import ChatGroq
    except Exception as exc:
        return (
            f"Groq support is not installed: {exc}\n\n"
            f"Fallback answer:\n{build_grounded_fallback_answer(query, context_blocks)}"
        )

    PROMPT = """You are MarketplaceGuide, an intelligent and helpful retrieval-augmented assistant specializing in "Top 11 Multi-Vendor Marketplace Platforms for eCommerce."

<domain_context>
The source document profiles 11 named multi-vendor marketplace platforms (such as VTEX, Mirakl, Yo!Kart, CS-Cart Multi-Vendor, Sharetribe, Marketplacer, Spryker, Adobe Commerce/Magento, BigCommerce, Arcadier, and WCFM Marketplace) and covers their deployment types, pricing/commission structures, vendor management, and enterprise suitability.
</domain_context>

<rules>
1. **Analyze Intent**: Address the core intent of the user's question. If the user asks open-ended, subjective, or comparative questions (like "which platform is best", "what are my options", "recommend a platform"), do NOT refuse. Instead, synthesize a comprehensive overview from the retrieved context, presenting the options along with their specific strengths, trade-offs, or use-cases (e.g., enterprise, startups, self-hosted vs. SaaS).
2. **Be Informative and Summarize**: Summarize the platforms in the best structured format (using bullet points, comparative summaries, or clear groupings) so the user gets a highly helpful response.
3. **Stay Grounded**: Answer using only the information available in the <retrieved_context> below. Do not make up facts or bring in external knowledge not mentioned in the context.
4. **Handle Off-Topic Queries Strict Refusal**: If the user's question is completely off-topic and has nothing to do with eCommerce, marketplace software, vendor management, platforms, or the context provided, respond exactly: "The provided document does not contain enough information to answer this question."
5. **Format & Tone**: Do not mention "the context", "retrieved sections", or "retrieval process". Answer directly with a professional, expert tone.

## Source Formatting Rule
At the end of your response, list the unique section paths from the context under a "Sources:" heading.
</rules>

<response_format>
Answer:
<answer>

Sources:
- <formatted source 1>
- <formatted source 2>
</response_format>

<retrieved_context>
{context}
</retrieved_context>

<user_question>
{query}
</user_question>

Answer:"""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key,
        )
        prompt = PROMPT.format(context=context, query=query)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return (
            f"Groq generation failed: {exc}\n\n"
            f"Fallback answer:\n{build_grounded_fallback_answer(query, context_blocks)}"
        )

    
def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval and answer generation for one or more queries.")
    parser.add_argument("queries", nargs="*", help="One or more questions to answer.")
    return parser.parse_args()


def main():
    args = parse_args()
    db_dir = os.path.join("data", "chroma_db")
    vector_db = load_vector_store(persist_dir=db_dir)

    queries = args.queries
    if not queries:
        print("No query provided. Enter one or more questions.")
        while True:
            user_query = input("Enter a query (press Enter to exit): ").strip()
            if not user_query:
                break
            queries.append(user_query)

    if not queries:
        print("No queries were provided. Exiting.")
        return

    for query in queries:
        print(f"\n Query: '{query}'")
        print("=" * 60)
        reranked = retrieve_and_rerank(vector_db, query, initial_k=10, top_n=3)

        for rank, (doc, score) in enumerate(reranked, start=1):
            print(f" Rank {rank} | Rerank Score: {score:.4f}")
            print(f"   Section: {doc.metadata.get('section_path')}")
            print(f"   \"{doc.page_content.strip()[:200]}...\"")
            print("-" * 60)

        answer = generate_answer(query, reranked)
        print(f"\n Generated Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
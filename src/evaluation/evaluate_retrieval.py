import sys
from pathlib import Path

# Add project root to path so we can import from src.retrieval
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.rerank_and_answer import load_vector_store, retrieve_and_rerank

# Define ground-truth test queries and the expected substring/section path.
# A query PASSES if any top-N chunk's section or content matches the target.
TEST_DATASET = [
    {
        "id": "q1",
        "query": "what is B2C business to consumer?",
        "expected_substring": "VTEX",  # Key string expected in context
        "expected_section": "Top 11 multi vendor marketplace platforms",
    },
    {
        "id": "q2",
        "query": "What are the main features and capabilities of marketplace software?",
        "expected_substring": "Key components include",
        "expected_section": "What are the types of marketplace software?",
    },
    {
        "id": "q3",
        "query": "How do commission fees or vendor management work across platforms?",
        "expected_substring": "commission",
        "expected_section": "Commission engine",
    },
]


def evaluate_retrieval(top_k_retrieve: int = 10, top_n_rerank: int = 3):
    """Evaluates the retrieval + reranking pipeline against ground truth expectations."""
    print(" Loading Chroma vector store...")
    vector_db = load_vector_store()

    passed_count = 0
    total_queries = len(TEST_DATASET)

    print(f"\n================ Running Retrieval Benchmark ================")
    print(f"Strategy: Similarity (K={top_k_retrieve}) -> Rerank (Top {top_n_rerank})\n")

    for test in TEST_DATASET:
        query_id = test["id"]
        query = test["query"]
        target_sub = test["expected_substring"].lower()
        target_sec = test["expected_section"].lower()

        print(f"[{query_id}] Query: '{query}'")

        # 1. Retrieve & Rerank
        reranked_results = retrieve_and_rerank(
            vector_db, query, initial_k=top_k_retrieve, top_n=top_n_rerank
        )

        # 2. Check if expected section or keyword was retrieved in Top-N
        is_passed = False
        matched_rank = None

        print("   Reranked Top Chunks:")
        for rank, (doc, score) in enumerate(reranked_results, start=1):
            sec_path = doc.metadata.get("section_path", "")
            content = doc.page_content

            print(f"     Rank {rank} | Score: {score:.4f} | Section: {sec_path}")

            # Match criteria: check either text content or section metadata path
            if target_sub in content.lower() or target_sec in sec_path.lower():
                is_passed = True
                if matched_rank is None:
                    matched_rank = rank

        # 3. Print PASS / FAIL result
        if is_passed:
            passed_count += 1
            print(f"   Status:  PASS (Matched target at Rank {matched_rank})\n")
        else:
            print(f"   Status: FAIL (Target '{test['expected_substring']}' not found in top {top_n_rerank})\n")

        print("-" * 65)

    # Summary Statistics
    hit_rate = (passed_count / total_queries) * 100
    print(f"\n EVALUATION SUMMARY")
    print(f"Passed: {passed_count}/{total_queries} queries")
    print(f"Hit Rate @ Top-{top_n_rerank}: {hit_rate:.1f}%")
    print("============================================================\n")


if __name__ == "__main__":
    evaluate_retrieval(top_k_retrieve=10, top_n_rerank=3)
"""
Lightweight routing agent for the eCommerce Marketplace RAG system.

This is a decision layer around the existing retrieval/generation pipeline
in src/retrieval/rerank_and_answer.py. It does NOT replace retrieval or
generation — it decides *how* to use them:

    1. classify the incoming query
    2. run a first-pass hybrid retrieval + rerank
    3. judge whether the evidence is strong enough
    4. if not, run a second, more targeted pass
    5. hand the best evidence to the existing answer generator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.retrieval.rerank_and_answer import (
    retrieve_and_rerank,
    generate_answer,
    rewrite_query,
)


class QueryIntent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    COMPARISON = "comparison"
    VAGUE = "vague"
    OFF_TOPIC = "off_topic"


COMPARISON_KEYWORDS = {
    "best", "top", "compare", "comparison", "vs", "versus",
    "which", "recommend", "better", "difference",
}

ATTRIBUTE_KEYWORDS = {
    "pricing", "price", "cost", "commission", "deployment",
    "self-hosted", "self hosted", "saas", "vendor management",
    "headless", "api", "enterprise",
}

DOMAIN_KEYWORDS = {
    "marketplace", "vendor", "ecommerce", "e-commerce", "platform",
    "vtex", "mirakl", "yo!kart", "yokart", "cs-cart", "sharetribe",
    "marketplacer", "spryker", "magento", "adobe commerce",
    "bigcommerce", "arcadier", "wcfm", "woocommerce", "commission",
    "multi-vendor", "multi vendor", "b2b", "b2c",
}

MIN_RERANK_SCORE = 0.15          # below this, evidence is considered weak
MIN_RESULTS_FOR_COMPARISON = 2   # comparisons need multiple distinct sections


def classify_intent(query: str) -> QueryIntent:
    q = query.lower().strip()

    if not q:
        return QueryIntent.VAGUE

    if any(k in q for k in COMPARISON_KEYWORDS):
        return QueryIntent.COMPARISON

    if any(k in q for k in DOMAIN_KEYWORDS) or any(k in q for k in ATTRIBUTE_KEYWORDS):
        return QueryIntent.FACT_LOOKUP

    if len(q.split()) <= 2:
        return QueryIntent.VAGUE

    # Let retrieval have the final say — the weak-evidence check below
    # will catch true off-topic questions even if the keyword pass misses.
    return QueryIntent.OFF_TOPIC


def _distinct_sections(reranked) -> int:
    return len({doc.metadata.get("section_path", "") for doc, _ in reranked})


def _evidence_is_weak(reranked, intent: QueryIntent) -> bool:
    if not reranked:
        return True
    if reranked[0][1] < MIN_RERANK_SCORE:
        return True
    if intent == QueryIntent.COMPARISON and _distinct_sections(reranked) < MIN_RESULTS_FOR_COMPARISON:
        return True
    return False


@dataclass
class AgentResult:
    query: str
    intent: QueryIntent
    reranked_chunks: list = field(default_factory=list)
    used_second_pass: bool = False
    answer: str = ""


def run_agent(vector_db, query: str, initial_k: int = 10, top_n: int = 3) -> AgentResult:
    """Route a single user query through classification, retrieval, an
    optional second retrieval pass, and generation."""
    intent = classify_intent(query)

    if intent == QueryIntent.VAGUE:
        return AgentResult(
            query=query,
            intent=intent,
            answer=(
                "Could you clarify your question a bit? For example, are you asking "
                "about a specific platform, pricing/commission structure, deployment "
                "type (SaaS vs self-hosted), or a comparison between platforms?"
            ),
        )

    # Comparisons benefit from a wider first net
    first_pass_k = initial_k * 2 if intent == QueryIntent.COMPARISON else initial_k
    reranked = retrieve_and_rerank(vector_db, query, initial_k=first_pass_k, top_n=top_n)

    used_second_pass = False
    if _evidence_is_weak(reranked, intent):
        used_second_pass = True
        variants = rewrite_query(query)
        broadened_query = " ".join(variants) if variants else query
        second_pass = retrieve_and_rerank(
            vector_db, broadened_query, initial_k=initial_k * 2, top_n=max(top_n, 5)
        )
        seen = {doc.page_content for doc, _ in reranked}
        merged = list(reranked)
        for doc, score in second_pass:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append((doc, score))
        merged.sort(key=lambda x: x[1], reverse=True)
        reranked = merged[: max(top_n, 5)]

    if not reranked:
        return AgentResult(
            query=query,
            intent=intent,
            used_second_pass=used_second_pass,
            answer="The provided document does not contain enough information to answer this question.",
        )

    answer = generate_answer(query, reranked)
    return AgentResult(
        query=query,
        intent=intent,
        reranked_chunks=reranked,
        used_second_pass=used_second_pass,
        answer=answer,
    )
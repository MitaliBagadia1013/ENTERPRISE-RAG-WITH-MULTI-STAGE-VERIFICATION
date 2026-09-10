import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import cohere

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings
from retrieval.semantic_retriever import RetrievalResult


@dataclass
class RerankedResult:
    chunk_id: str
    text: str
    semantic_score: float
    rerank_score: float
    contract_id: str
    contract_type: str
    rbac_roles: list[str]
    chunk_index: int
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RerankingMetrics:
    query: str
    input_count: int
    output_count: int
    rerank_time_ms: float
    model: str
    cost_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


class CohereReranker:
    PRICE_PER_SEARCH = 2.0 / 1000000

    def __init__(self, model: str = "rerank-english-v3.0"):
        print("Initializing Cohere Reranker...")
        if not settings.COHERE_API_KEY:
            raise ValueError(
                "COHERE_API_KEY not found in environment variables. Get your key from: https://dashboard.cohere.com/api-keys"
            )
        self.client = cohere.Client(api_key=settings.COHERE_API_KEY)
        self.model = model
        print("Cohere client ready")
        print(f"Model: {model}")
        print(f"Cost: ${self.PRICE_PER_SEARCH * 1000:.6f} per 1000 searches")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> tuple[list[RerankedResult], RerankingMetrics]:
        if not results:
            empty_metrics = RerankingMetrics(
                query=query,
                input_count=0,
                output_count=0,
                rerank_time_ms=0.0,
                model=self.model,
                cost_usd=0.0,
            )
            return ([], empty_metrics)
        documents = [result.text for result in results]
        start_time = time.time()
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=False,
            )
        except Exception as e:
            print(f"Cohere rerank failed: {e}")
            raise
        rerank_time_ms = (time.time() - start_time) * 1000
        reranked_results = []
        for rank_result in response.results:
            original_idx = rank_result.index
            original_result = results[original_idx]
            rerank_score = rank_result.relevance_score
            if rerank_score < min_score:
                continue
            reranked = RerankedResult(
                chunk_id=original_result.chunk_id,
                text=original_result.text,
                semantic_score=original_result.score,
                rerank_score=rerank_score,
                contract_id=original_result.contract_id,
                contract_type=original_result.contract_type,
                rbac_roles=original_result.rbac_roles,
                chunk_index=original_result.chunk_index,
                word_count=original_result.word_count,
            )
            reranked_results.append(reranked)
        num_searches = len(documents)
        cost_usd = num_searches * self.PRICE_PER_SEARCH
        metrics = RerankingMetrics(
            query=query,
            input_count=len(results),
            output_count=len(reranked_results),
            rerank_time_ms=rerank_time_ms,
            model=self.model,
            cost_usd=cost_usd,
        )
        return (reranked_results, metrics)

    def rerank_with_semantic_retriever(
        self,
        semantic_retriever,
        query: str,
        user_role: str | None = None,
        semantic_top_k: int = 20,
        rerank_top_k: int = 5,
        min_rerank_score: float = 0.0,
        contract_types: list[str] | None = None,
    ) -> tuple[list[RerankedResult], RerankingMetrics]:
        semantic_results = semantic_retriever.search(
            query=query,
            user_role=user_role,
            top_k=semantic_top_k,
            contract_types=contract_types,
        )
        if not semantic_results:
            print(f"No semantic results found for query: {query}")
            empty_metrics = RerankingMetrics(
                query=query,
                input_count=0,
                output_count=0,
                rerank_time_ms=0.0,
                model=self.model,
                cost_usd=0.0,
            )
            return ([], empty_metrics)
        reranked_results, metrics = self.rerank(
            query=query,
            results=semantic_results,
            top_k=rerank_top_k,
            min_score=min_rerank_score,
        )
        return (reranked_results, metrics)


def format_reranked_results(
    results: list[RerankedResult], metrics: RerankingMetrics
) -> str:
    if not results:
        return "No results found after reranking."
    output = []
    output.append(f"\n{'=' * 70}")
    output.append(f"Reranked Results ({len(results)} chunks)")
    output.append(f"{'=' * 70}")
    output.append(f"Query: {metrics.query}")
    output.append(
        f"Input: {metrics.input_count} chunks Output: {metrics.output_count} chunks"
    )
    output.append(f"Time: {metrics.rerank_time_ms:.1f}ms")
    output.append(f"Cost: ${metrics.cost_usd:.6f}")
    output.append(f"Model: {metrics.model}")
    output.append(f"{'=' * 70}\n")
    for i, result in enumerate(results, 1):
        score_delta = result.rerank_score - result.semantic_score
        score_change = (
            f"+{score_delta:.3f}" if score_delta > 0 else f"{score_delta:.3f}"
        )
        output.append(
            f"[{i}] Rerank Score: {result.rerank_score:.4f} (semantic: {result.semantic_score:.4f}, Δ{score_change})"
        )
        output.append(f"Contract: {result.contract_id} ({result.contract_type})")
        output.append(f"Chunk: {result.chunk_index}")
        output.append(f"Roles: {', '.join(result.rbac_roles)}")
        output.append(f"Text: {result.text[:200]}...")
        output.append("")
    return "\n".join(output)


def compare_rankings(
    semantic_results: list[RetrievalResult], reranked_results: list[RerankedResult]
) -> str:
    output = []
    output.append(f"\n{'=' * 70}")
    output.append("Semantic vs Reranked Comparison")
    output.append(f"{'=' * 70}\n")
    semantic_ranks = {
        result.chunk_id: (i + 1, result.score)
        for i, result in enumerate(semantic_results)
    }
    for i, reranked in enumerate(reranked_results, 1):
        chunk_id = reranked.chunk_id
        semantic_rank, semantic_score = semantic_ranks.get(chunk_id, (None, None))
        if semantic_rank:
            rank_change = semantic_rank - i
            if rank_change > 0:
                change_str = f"{rank_change}"
            elif rank_change < 0:
                change_str = f"{abs(rank_change)}"
            else:
                change_str = "="
            output.append(
                f"[{i:2d}] {change_str:>4s} (was #{semantic_rank:2d}) | Rerank: {reranked.rerank_score:.3f} | Semantic: {semantic_score:.3f} | {chunk_id}"
            )
    return "\n".join(output)


if __name__ == "__main__":
    print("=" * 70)
    print("Cohere Reranker - Test")
    print("=" * 70)
    from retrieval.semantic_retriever import SemanticRetriever

    retriever = SemanticRetriever()
    reranker = CohereReranker()
    query = "What are the termination clauses in the contract?"
    print(f"\nQuery: {query}")
    print("\nStep 1: Semantic Retrieval (fetching top-20)...")
    semantic_results = retriever.search(query=query, user_role=None, top_k=20)
    print(f"Found {len(semantic_results)} semantic results")
    print("\nStep 2: Reranking (narrowing to top-5)...")
    reranked_results, metrics = reranker.rerank(
        query=query, results=semantic_results, top_k=5
    )
    print(format_reranked_results(reranked_results, metrics))
    print(compare_rankings(semantic_results[:10], reranked_results))
    print("\n" + "=" * 70)
    print("End-to-End Pipeline Test (Semantic + Rerank)")
    print("=" * 70)
    e2e_results, e2e_metrics = reranker.rerank_with_semantic_retriever(
        semantic_retriever=retriever,
        query="What happens if the agreement is terminated early?",
        user_role=None,
        semantic_top_k=20,
        rerank_top_k=5,
    )
    print(format_reranked_results(e2e_results, e2e_metrics))

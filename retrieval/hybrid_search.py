import re


class HybridSearchScorer:

    def __init__(self, semantic_weight: float = 0.6, keyword_weight: float = 0.4):
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "what",
            "when",
            "where",
            "who",
            "how",
            "shall",
            "may",
            "hereby",
            "whereas",
            "thereof",
        }

    def rerank_with_keywords(self, query: str, results: list[dict]) -> list[dict]:
        if not results:
            return results
        query_keywords = self._extract_keywords(query)
        keyword_scores = []
        for result in results:
            text = result.text if hasattr(result, "text") else result.get("text", "")
            score = self._calculate_keyword_score(query_keywords, text)
            keyword_scores.append(score)
        if max(keyword_scores) > 0:
            keyword_scores = [s / max(keyword_scores) for s in keyword_scores]
        hybrid_results = []
        for i, result in enumerate(results):
            semantic_score = (
                result.score if hasattr(result, "score") else result.get("score", 0.0)
            )
            hybrid_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_scores[i]
            )
            if hasattr(result, "score"):
                result_dict = (
                    result.to_dict()
                    if hasattr(result, "to_dict")
                    else {
                        "chunk_id": result.chunk_id,
                        "text": result.text,
                        "score": hybrid_score,
                        "contract_id": result.contract_id,
                        "contract_type": result.contract_type,
                        "rbac_roles": result.rbac_roles,
                        "chunk_index": result.chunk_index,
                        "word_count": result.word_count,
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_scores[i],
                    }
                )
                result_dict["score"] = hybrid_score
                result_dict["semantic_score"] = semantic_score
                result_dict["keyword_score"] = keyword_scores[i]
                hybrid_results.append(result_dict)
            else:
                result_copy = result.copy()
                result_copy["score"] = hybrid_score
                result_copy["semantic_score"] = semantic_score
                result_copy["keyword_score"] = keyword_scores[i]
                hybrid_results.append(result_copy)
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results

    def _extract_keywords(self, text: str) -> set[str]:
        text = text.lower()
        text = re.sub("[^\\w\\s]", "", text)
        words = text.split()
        keywords = {
            word for word in words if word not in self.stopwords and len(word) > 2
        }
        return keywords

    def _calculate_keyword_score(
        self, query_keywords: set[str], document_text: str
    ) -> float:
        if not query_keywords:
            return 0.0
        doc_keywords = self._extract_keywords(document_text)
        overlap = query_keywords & doc_keywords
        if not overlap:
            return 0.0
        doc_words = document_text.lower().split()
        doc_length = len(doc_words)
        score = 0.0
        for keyword in overlap:
            tf = doc_words.count(keyword)
            k1 = 1.5
            normalized_tf = tf * (k1 + 1) / (tf + k1)
            score += normalized_tf
        coverage_bonus = len(overlap) / len(query_keywords)
        score *= coverage_bonus
        return score


class ReciprocalRankFusion:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_lists: list[list[dict]], top_k: int = 10) -> list[dict]:
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                chunk_id = result.get("chunk_id") or (
                    result.chunk_id if hasattr(result, "chunk_id") else None
                )
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = {"score": 0.0, "result": result}
                rrf_scores[chunk_id]["score"] += 1.0 / (self.k + rank + 1)
        fused_results = sorted(
            rrf_scores.values(), key=lambda x: x["score"], reverse=True
        )
        return [item["result"] for item in fused_results[:top_k]]


if __name__ == "__main__":
    semantic_results = [
        {
            "chunk_id": "chunk_1",
            "text": "The termination clause allows either party to end the agreement with 30 days notice.",
            "score": 0.85,
            "contract_id": "C001",
        },
        {
            "chunk_id": "chunk_2",
            "text": "Payment shall be made within 15 business days of invoice receipt.",
            "score": 0.78,
            "contract_id": "C001",
        },
        {
            "chunk_id": "chunk_3",
            "text": "Upon termination, all confidential information must be returned immediately.",
            "score": 0.72,
            "contract_id": "C002",
        },
    ]
    query = "termination conditions and notice period"
    print("=" * 70)
    print("HYBRID SEARCH EXAMPLE")
    print("=" * 70)
    print(f"\nQuery: {query}\n")
    scorer = HybridSearchScorer(semantic_weight=0.6, keyword_weight=0.4)
    reranked = scorer.rerank_with_keywords(query, semantic_results)
    print("RERANKED RESULTS:\n")
    for i, result in enumerate(reranked, 1):
        print(
            f"{i}. Score: {result['score']:.3f} (semantic: {result['semantic_score']:.3f}, keyword: {result['keyword_score']:.3f})"
        )
        print(f"Text: {result['text'][:100]}...")
        print()
    print("=" * 70)
    print("Notice how chunk_3 (termination + return) got boosted")
    print("despite having lower semantic score!")
    print("=" * 70)

import sys
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    contract_id: str
    contract_type: str
    rbac_roles: list[str]
    chunk_index: int
    word_count: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "contract_id": self.contract_id,
            "contract_type": self.contract_type,
            "rbac_roles": self.rbac_roles,
            "chunk_index": self.chunk_index,
            "word_count": self.word_count,
        }


class SemanticRetriever:

    def __init__(self):
        print("Initializing Semantic Retriever...")
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pinecone_client.Index(settings.PINECONE_INDEX_NAME)
        stats = self.index.describe_index_stats()
        vector_count = stats.total_vector_count
        if vector_count == 0:
            raise ValueError(
                f"Pinecone index '{settings.PINECONE_INDEX_NAME}' is empty! Run the embedder first."
            )
        print(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
        print(f"Index contains {vector_count:,} vectors")
        print("OpenAI client ready")

    def embed_query(self, query: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            input=query, model=settings.EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        user_role: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        contract_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embed_query(query)
        metadata_filter = self._build_metadata_filter(
            user_role=user_role, contract_types=contract_types
        )
        fetch_k = top_k * 3 if metadata_filter else top_k
        pinecone_results = self.index.query(
            vector=query_embedding,
            top_k=fetch_k,
            include_metadata=True,
            filter=metadata_filter,
        )
        results = self._parse_results(
            pinecone_results, min_score=min_score, top_k=top_k
        )
        return results

    def _build_metadata_filter(
        self, user_role: str | None = None, contract_types: list[str] | None = None
    ) -> dict | None:
        filters = {}
        if user_role:
            filters["rbac_roles"] = {"$in": [user_role, "all"]}
        if contract_types:
            filters["contract_type"] = {"$in": contract_types}
        if not filters:
            return None
        if len(filters) == 1:
            return filters
        return {"$and": [{key: value} for key, value in filters.items()]}

    def _parse_results(
        self, pinecone_results, min_score: float = 0.0, top_k: int = 5
    ) -> list[RetrievalResult]:
        results = []
        for match in pinecone_results.matches:
            if match.score < min_score:
                continue
            metadata = match.metadata
            result = RetrievalResult(
                chunk_id=match.id,
                text=metadata.get("text", ""),
                score=float(match.score),
                contract_id=metadata.get("contract_id", ""),
                contract_type=metadata.get("contract_type", ""),
                rbac_roles=metadata.get("rbac_roles", []),
                chunk_index=metadata.get("chunk_index", 0),
                word_count=metadata.get("word_count", 0),
            )
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    def get_context_window(
        self, chunk_id: str, window_size: int = 2
    ) -> list[RetrievalResult]:
        parts = chunk_id.split("_chunk_")
        if len(parts) != 2:
            return []
        contract_prefix = parts[0]
        center_index = int(parts[1])
        filter_dict = {"contract_id": contract_prefix}
        results = self.index.query(
            vector=[0.0] * settings.EMBEDDING_DIMENSION,
            top_k=100,
            include_metadata=True,
            filter=filter_dict,
        )
        window_chunks = []
        min_index = center_index - window_size
        max_index = center_index + window_size
        for match in results.matches:
            chunk_idx = match.metadata.get("chunk_index", -1)
            if min_index <= chunk_idx <= max_index:
                result = RetrievalResult(
                    chunk_id=match.id,
                    text=match.metadata.get("text", ""),
                    score=float(match.score),
                    contract_id=match.metadata.get("contract_id", ""),
                    contract_type=match.metadata.get("contract_type", ""),
                    rbac_roles=match.metadata.get("rbac_roles", []),
                    chunk_index=chunk_idx,
                    word_count=match.metadata.get("word_count", 0),
                )
                window_chunks.append(result)
        window_chunks.sort(key=lambda x: x.chunk_index)
        return window_chunks

    def search_by_contract_id(
        self, contract_id: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        query_embedding = self.embed_query(query)
        filter_dict = {"contract_id": contract_id}
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )
        return self._parse_results(results, top_k=top_k)


def format_results(results: list[RetrievalResult]) -> str:
    if not results:
        return "No results found."
    output = []
    output.append(f"\n{'=' * 70}")
    output.append(f"Found {len(results)} relevant chunks:")
    output.append(f"{'=' * 70}\n")
    for i, result in enumerate(results, 1):
        output.append(f"[{i}] Score: {result.score:.4f}")
        output.append(f"Contract: {result.contract_id} ({result.contract_type})")
        output.append(f"Chunk: {result.chunk_index}")
        output.append(f"Roles: {', '.join(result.rbac_roles)}")
        output.append(f"Text: {result.text[:200]}...")
        output.append("")
    return "\n".join(output)


if __name__ == "__main__":
    print("=" * 70)
    print("Semantic Retriever - Test")
    print("=" * 70)
    retriever = SemanticRetriever()
    query = "What are the termination clauses in the contract?"
    print(f"\nQuery: {query}")
    print("\nSearching as 'legal_team' role...")
    results = retriever.search(query=query, user_role="legal_team", top_k=3)
    print(format_results(results))
    print("\nSearching as admin (no RBAC filtering)...")
    admin_results = retriever.search(query=query, user_role=None, top_k=3)
    print(format_results(admin_results))

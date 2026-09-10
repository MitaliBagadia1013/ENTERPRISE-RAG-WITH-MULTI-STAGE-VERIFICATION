from functools import lru_cache
import cohere
import redis
from openai import OpenAI
from pinecone import Pinecone
from cache.semantic_cache import SemanticCache
from config.settings import settings
from monitoring.langfuse_monitor import LangfuseMonitor
from retrieval.enhanced_retriever import EnhancedRetriever
from retrieval.reranker import CohereReranker
from retrieval.semantic_retriever import SemanticRetriever
from verification.answer_verifier import AnswerVerifier
from verification.self_correction_agent import SelfCorrectionAgent


@lru_cache
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_cohere_client() -> cohere.Client:
    return cohere.Client(api_key=settings.COHERE_API_KEY)


@lru_cache
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY)


@lru_cache
def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
    )


@lru_cache
def get_semantic_cache() -> SemanticCache:
    return SemanticCache(
        redis_host=settings.REDIS_HOST,
        redis_port=settings.REDIS_PORT,
        redis_db=settings.REDIS_DB,
        similarity_threshold=0.92,
        ttl_hours=24,
    )


@lru_cache
def get_semantic_retriever() -> SemanticRetriever:
    return SemanticRetriever()


@lru_cache
def get_enhanced_retriever() -> EnhancedRetriever:
    return EnhancedRetriever(
        use_query_expansion=True,
        use_hybrid_search=True,
        use_cohere_reranking=True,
        semantic_weight=0.6,
        keyword_weight=0.4,
    )


@lru_cache
def get_reranker() -> CohereReranker:
    return CohereReranker(model=settings.COHERE_RERANK_MODEL)


@lru_cache
def get_self_correction_agent() -> SelfCorrectionAgent:
    return SelfCorrectionAgent(
        model=settings.OPENAI_MODEL, max_iterations=3, quality_threshold=85
    )


@lru_cache
def get_answer_verifier() -> AnswerVerifier:
    return AnswerVerifier()


@lru_cache
def get_langfuse_monitor() -> LangfuseMonitor:
    return LangfuseMonitor()


class RAGPipeline:

    def __init__(
        self,
        cache: SemanticCache,
        retriever: SemanticRetriever,
        reranker: CohereReranker,
        self_correction: SelfCorrectionAgent,
        verifier: AnswerVerifier,
    ):
        self.cache = cache
        self.retriever = retriever
        self.reranker = reranker
        self.self_correction = self_correction
        self.verifier = verifier

    async def query(
        self,
        query: str,
        access_level: str,
        top_k: int = 10,
        use_cache: bool = True,
        rerank: bool = True,
        verify: bool = True,
    ) -> dict:
        import time

        start_time = time.time()
        total_cost = 0.0
        cache_hit = False
        cache_similarity = None
        if use_cache:
            cached_result = self.cache.get(query, access_level)
            if cached_result:
                cache_hit = True
                cache_similarity = cached_result.get("cache_similarity")
                elapsed_ms = (time.time() - start_time) * 1000
                return {
                    **cached_result,
                    "cache_hit": True,
                    "cache_similarity": cache_similarity,
                    "total_cost": 0.0,
                    "latency_ms": int(elapsed_ms),
                }
        chunks = self.retriever.search(
            query=query, top_k=top_k, access_level=access_level
        )
        total_cost += 0.001
        if rerank and chunks:
            reranked = self.reranker.rerank(
                query=query, chunks=chunks, top_k=min(5, len(chunks))
            )
            chunks = reranked["chunks"]
            total_cost += reranked.get("cost", 4e-05)
        answer_result = self.self_correction.generate_answer(query=query, chunks=chunks)
        answer = answer_result["answer"]
        total_cost += answer_result.get("cost", 0.025)
        verification = None
        if verify:
            verification = self.verifier.verify_answer(
                query=query, answer=answer, source_chunks=chunks
            )
            total_cost += verification.get("cost", 0.013)
        if use_cache and verification and verification["is_trustworthy"]:
            self.cache.set(
                query=query,
                answer=answer,
                source_chunks=chunks,
                verification_result=verification,
                access_level=access_level,
                generation_cost=total_cost,
            )
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "query": query,
            "answer": answer,
            "sources": chunks,
            "verification": verification,
            "cache_hit": False,
            "cache_similarity": None,
            "total_cost": total_cost,
            "latency_ms": int(elapsed_ms),
            "metadata": {
                "num_chunks_retrieved": top_k,
                "num_chunks_reranked": len(chunks) if rerank else 0,
            },
        }


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        cache=get_semantic_cache(),
        retriever=get_semantic_retriever(),
        reranker=get_reranker(),
        self_correction=get_self_correction_agent(),
        verifier=get_answer_verifier(),
    )


async def check_redis_health() -> dict:
    import time

    try:
        start = time.time()
        redis_client = get_redis_client()
        redis_client.ping()
        elapsed_ms = (time.time() - start) * 1000
        return {
            "service": "redis",
            "status": "healthy",
            "response_time_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "service": "redis",
            "status": "unhealthy",
            "response_time_ms": None,
            "error": str(e),
        }


async def check_pinecone_health() -> dict:
    import time

    try:
        start = time.time()
        pc = get_pinecone_client()
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        elapsed_ms = (time.time() - start) * 1000
        return {
            "service": "pinecone",
            "status": "healthy",
            "response_time_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "service": "pinecone",
            "status": "unhealthy",
            "response_time_ms": None,
            "error": str(e),
        }


async def check_openai_health() -> dict:
    import time

    try:
        start = time.time()
        client = get_openai_client()
        client.models.list()
        elapsed_ms = (time.time() - start) * 1000
        return {
            "service": "openai",
            "status": "healthy",
            "response_time_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "service": "openai",
            "status": "unhealthy",
            "response_time_ms": None,
            "error": str(e),
        }


async def check_cohere_health() -> dict:
    import time

    try:
        start = time.time()
        client = get_cohere_client()
        client.check_api_key()
        elapsed_ms = (time.time() - start) * 1000
        return {
            "service": "cohere",
            "status": "healthy",
            "response_time_ms": round(elapsed_ms, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "service": "cohere",
            "status": "unhealthy",
            "response_time_ms": None,
            "error": str(e),
        }


def cleanup_resources():
    try:
        redis_client = get_redis_client()
        redis_client.close()
        print("Redis connection closed")
    except Exception as e:
        print(f"Error closing Redis: {e}")

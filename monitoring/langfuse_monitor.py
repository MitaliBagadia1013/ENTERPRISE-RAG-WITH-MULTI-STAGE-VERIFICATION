from typing import Any
from langfuse import Langfuse
from config.settings import settings


class LangfuseMonitor:

    def __init__(self):
        print("Initializing Langfuse Monitor...")
        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            print("Langfuse not configured - monitoring disabled")
            print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")
            print("Get keys from: https://cloud.langfuse.com")
            self.enabled = False
            self.client = None
            return
        try:
            self.client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            self.enabled = True
            print("Langfuse client ready")
            print(f"Host: {settings.LANGFUSE_HOST}")
        except Exception as e:
            print(f"Failed to initialize Langfuse: {e}")
            self.enabled = False
            self.client = None

    def start_trace(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        try:
            trace_id = self.client.create_trace_id()
            self.client.create_event(
                trace_id=trace_id,
                name="rag_query_start",
                user_id=user_id,
                session_id=session_id,
                input={"query": query},
                metadata={
                    **(metadata or {}),
                    "tags": ["production", "rag", settings.ENVIRONMENT],
                },
            )
            return trace_id
        except Exception as e:
            print(f"Failed to start trace: {e}")
            return None

    def log_cache_check(
        self,
        trace_id: str,
        query: str,
        cache_hit: bool,
        cached_answer: str | None = None,
        similarity_score: float | None = None,
        latency_ms: float = 0,
    ):
        if not self.enabled or not trace_id:
            return
        try:
            self.client.create_event(
                trace_id=trace_id,
                name="cache_check",
                input={"query": query},
                output={
                    "cache_hit": cache_hit,
                    "answer": cached_answer if cache_hit else None,
                    "similarity_score": similarity_score,
                },
                metadata={
                    "latency_ms": latency_ms,
                    "cost_saved": 0.018 if cache_hit else 0,
                },
            )
        except Exception as e:
            print(f"Failed to log cache check: {e}")

    def log_retrieval(
        self,
        trace_id: str,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
        latency_ms: float,
        cost_usd: float = 0,
    ):
        if not self.enabled or not trace_id:
            return
        try:
            chunk_metadata = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("score"),
                    "contract_id": c.get("contract_id"),
                }
                for c in chunks[:5]
            ]
            self.client.span(
                trace_id=trace_id,
                name="semantic_retrieval",
                input={"query": query, "top_k": top_k},
                output={"num_chunks": len(chunks), "chunks": chunk_metadata},
                metadata={
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "vector_db": "pinecone",
                    "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                },
                level="DEFAULT",
                usage={"input": 1, "output": 0, "unit": "TOKENS"},
            )
        except Exception as e:
            print(f"Failed to log retrieval: {e}")

    def log_reranking(
        self,
        trace_id: str,
        query: str,
        input_chunks: list[dict[str, Any]],
        output_chunks: list[dict[str, Any]],
        latency_ms: float,
        cost_usd: float,
    ):
        if not self.enabled or not trace_id:
            return
        try:
            self.client.span(
                trace_id=trace_id,
                name="reranking",
                input={"query": query, "num_input_chunks": len(input_chunks)},
                output={
                    "num_output_chunks": len(output_chunks),
                    "top_chunks": [
                        {
                            "chunk_id": c.get("chunk_id"),
                            "rerank_score": c.get("rerank_score"),
                        }
                        for c in output_chunks[:3]
                    ],
                },
                metadata={
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "model": settings.COHERE_RERANK_MODEL,
                },
                level="DEFAULT",
            )
        except Exception as e:
            print(f"Failed to log reranking: {e}")

    def log_generation(
        self,
        trace_id: str,
        query: str,
        context_chunks: list[dict[str, Any]],
        answer: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
        num_iterations: int = 1,
    ):
        if not self.enabled or not trace_id:
            return
        try:
            total_tokens = prompt_tokens + completion_tokens
            self.client.generation(
                trace_id=trace_id,
                name="answer_generation",
                input={"query": query, "num_context_chunks": len(context_chunks)},
                output=answer,
                model=model,
                usage={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": total_tokens,
                    "unit": "TOKENS",
                },
                metadata={
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "num_iterations": num_iterations,
                    "self_corrected": num_iterations > 1,
                },
                level="DEFAULT",
            )
        except Exception as e:
            print(f"Failed to log generation: {e}")

    def log_verification(
        self,
        trace_id: str,
        answer: str,
        verification_result: dict[str, Any],
        latency_ms: float,
        cost_usd: float = 0,
    ):
        if not self.enabled or not trace_id:
            return
        try:
            self.client.span(
                trace_id=trace_id,
                name="answer_verification",
                input={"answer": answer},
                output=verification_result,
                metadata={
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "verification_model": "gpt-4",
                },
                level="DEFAULT",
            )
        except Exception as e:
            print(f"Failed to log verification: {e}")

    def end_trace(
        self,
        trace_id: str,
        success: bool,
        total_cost_usd: float,
        total_latency_ms: float,
        answer: str | None = None,
        error: str | None = None,
    ):
        if not self.enabled or not trace_id:
            return
        try:
            self.client.trace(
                id=trace_id,
                output={"success": success, "answer": answer, "error": error},
                metadata={
                    "total_cost_usd": total_cost_usd,
                    "total_latency_ms": total_latency_ms,
                    "cost_per_second": (
                        total_cost_usd / total_latency_ms * 1000
                        if total_latency_ms > 0
                        else 0
                    ),
                },
            )
        except Exception as e:
            print(f"Failed to end trace: {e}")

    def log_user_feedback(
        self, trace_id: str, score: float, comment: str | None = None
    ):
        if not self.enabled or not trace_id:
            return
        try:
            self.client.score(
                trace_id=trace_id, name="user_feedback", value=score, comment=comment
            )
        except Exception as e:
            print(f"Failed to log user feedback: {e}")

    def flush(self):
        if self.enabled and self.client:
            try:
                self.client.flush()
                print("Langfuse events flushed")
            except Exception as e:
                print(f"Failed to flush Langfuse: {e}")


_monitor_instance = None


def get_monitor() -> LangfuseMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = LangfuseMonitor()
    return _monitor_instance

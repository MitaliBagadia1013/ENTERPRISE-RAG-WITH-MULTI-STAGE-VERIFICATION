import hashlib
import json
from datetime import datetime, timedelta
from typing import Any
import numpy as np
import redis
from openai import OpenAI
from config.settings import settings


class SemanticCache:

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        similarity_threshold: float = 0.92,
        ttl_hours: int = 24,
    ):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=False
        )
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-large"
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600
        self.cache_hits = 0
        self.cache_misses = 0
        self.cost_saved = 0.0
        self.QUERY_EMBEDDING_PREFIX = "query_emb:"
        self.ANSWER_PREFIX = "answer:"
        self.METADATA_PREFIX = "meta:"
        self.INDEX_KEY = "query_index"

    def _generate_embedding(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return response.data[0].embedding

    def _compute_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        dot_product = np.dot(emb1_np, emb2_np)
        norm1 = np.linalg.norm(emb1_np)
        norm2 = np.linalg.norm(emb2_np)
        return dot_product / (norm1 * norm2)

    def _generate_cache_key(self, query: str, access_level: str) -> str:
        combined = f"{query}|{access_level}"
        hash_key = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return hash_key

    def get(self, query: str, access_level: str = "user") -> dict[str, Any] | None:
        query_embedding = self._generate_embedding(query)
        all_keys = self.redis_client.keys(f"{self.QUERY_EMBEDDING_PREFIX}*")
        best_match = None
        best_similarity = 0.0
        for key in all_keys:
            cache_key = key.decode().replace(self.QUERY_EMBEDDING_PREFIX, "")
            meta_key = f"{self.METADATA_PREFIX}{cache_key}"
            meta_data = self.redis_client.get(meta_key)
            if not meta_data:
                continue
            meta = json.loads(meta_data.decode())
            if meta.get("access_level") != access_level:
                continue
            cached_embedding_data = self.redis_client.get(key)
            if not cached_embedding_data:
                continue
            cached_embedding = json.loads(cached_embedding_data.decode())
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cache_key
        if best_match and best_similarity >= self.similarity_threshold:
            answer_key = f"{self.ANSWER_PREFIX}{best_match}"
            answer_data = self.redis_client.get(answer_key)
            if answer_data:
                self.cache_hits += 1
                cached_answer = json.loads(answer_data.decode())
                meta_key = f"{self.METADATA_PREFIX}{best_match}"
                meta_data = self.redis_client.get(meta_key)
                if meta_data:
                    meta = json.loads(meta_data.decode())
                    self.cost_saved += meta.get("generation_cost", 0.0)
                cached_answer["from_cache"] = True
                cached_answer["cache_similarity"] = best_similarity
                cached_answer["original_query"] = cached_answer.get("query")
                return cached_answer
        self.cache_misses += 1
        return None

    def set(
        self,
        query: str,
        answer: str,
        source_chunks: list[dict[str, Any]],
        verification_result: dict[str, Any],
        access_level: str = "user",
        generation_cost: float = 0.0,
    ) -> None:
        if not verification_result.get("is_trustworthy", False):
            return
        cache_key = self._generate_cache_key(query, access_level)
        query_embedding = self._generate_embedding(query)
        cache_data = {
            "query": query,
            "answer": answer,
            "source_chunks": source_chunks,
            "verification_result": verification_result,
            "cached_at": datetime.now().isoformat(),
            "access_level": access_level,
        }
        metadata = {
            "query": query,
            "access_level": access_level,
            "cached_at": datetime.now().isoformat(),
            "generation_cost": generation_cost,
            "overall_score": verification_result.get("overall_score", 0.0),
        }
        embedding_key = f"{self.QUERY_EMBEDDING_PREFIX}{cache_key}"
        answer_key = f"{self.ANSWER_PREFIX}{cache_key}"
        meta_key = f"{self.METADATA_PREFIX}{cache_key}"
        self.redis_client.setex(
            embedding_key, self.ttl_seconds, json.dumps(query_embedding)
        )
        self.redis_client.setex(answer_key, self.ttl_seconds, json.dumps(cache_data))
        self.redis_client.setex(meta_key, self.ttl_seconds, json.dumps(metadata))

    def invalidate(self, query: str, access_level: str = "user") -> bool:
        cache_key = self._generate_cache_key(query, access_level)
        embedding_key = f"{self.QUERY_EMBEDDING_PREFIX}{cache_key}"
        answer_key = f"{self.ANSWER_PREFIX}{cache_key}"
        meta_key = f"{self.METADATA_PREFIX}{cache_key}"
        deleted = self.redis_client.delete(embedding_key, answer_key, meta_key)
        return deleted > 0

    def clear(self) -> int:
        keys_to_delete = []
        keys_to_delete.extend(self.redis_client.keys(f"{self.QUERY_EMBEDDING_PREFIX}*"))
        keys_to_delete.extend(self.redis_client.keys(f"{self.ANSWER_PREFIX}*"))
        keys_to_delete.extend(self.redis_client.keys(f"{self.METADATA_PREFIX}*"))
        if keys_to_delete:
            return self.redis_client.delete(*keys_to_delete)
        return 0

    def get_metrics(self) -> dict[str, Any]:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests * 100 if total_requests > 0 else 0.0
        cache_size = len(self.redis_client.keys(f"{self.ANSWER_PREFIX}*"))
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cost_saved": self.cost_saved,
            "cache_size": cache_size,
            "similarity_threshold": self.similarity_threshold,
        }

    def get_cache_info(self) -> list[dict[str, Any]]:
        cache_info = []
        meta_keys = self.redis_client.keys(f"{self.METADATA_PREFIX}*")
        for meta_key in meta_keys:
            meta_data = self.redis_client.get(meta_key)
            if meta_data:
                meta = json.loads(meta_data.decode())
                ttl = self.redis_client.ttl(meta_key)
                meta["ttl_seconds"] = ttl
                meta["expires_at"] = (
                    (datetime.now() + timedelta(seconds=ttl)).isoformat()
                    if ttl > 0
                    else None
                )
                cache_info.append(meta)
        return cache_info


def create_cache(
    similarity_threshold: float = 0.92, ttl_hours: int = 24
) -> SemanticCache:
    return SemanticCache(similarity_threshold=similarity_threshold, ttl_hours=ttl_hours)

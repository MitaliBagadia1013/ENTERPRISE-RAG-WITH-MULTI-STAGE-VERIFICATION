from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, validator


class AccessLevel(str, Enum):
    ADMIN = "admin"
    USER = "user"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    class Config:
        json_schema_extra = {"example": {"username": "admin", "password": "admin123"}}


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    access_level: AccessLevel
    expires_in: int = 3600

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "access_level": "admin",
                "expires_in": 3600,
            }
        }


class TokenPayload(BaseModel):
    sub: str
    access_level: AccessLevel
    exp: datetime


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="User question")
    use_cache: bool = Field(default=True, description="Enable semantic cache")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )
    rerank: bool = Field(default=True, description="Enable Cohere reranking")
    verify: bool = Field(default=True, description="Enable 5-stage verification")

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the termination clauses in the contract?",
                "use_cache": True,
                "top_k": 5,
                "rerank": True,
                "verify": True,
            }
        }


class SourceChunk(BaseModel):
    chunk_id: str
    contract_id: str
    content: str
    relevance_score: float
    rerank_score: float | None = None
    page_number: int | None = None


class VerificationResult(BaseModel):
    is_trustworthy: bool
    overall_score: float
    completeness_score: float
    contradiction_score: float
    hallucination_score: float
    citation_score: float
    reasoning: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunk]
    verification: VerificationResult
    metadata: dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False
    cache_similarity: float | None = None
    total_cost: float = 0.0
    latency_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the termination clauses?",
                "answer": "The contract requires 90-day written notice...",
                "sources": [
                    {
                        "chunk_id": "chunk_123",
                        "contract_id": "cuad_0001",
                        "content": "Either party may terminate...",
                        "relevance_score": 0.92,
                        "rerank_score": 0.95,
                    }
                ],
                "verification": {
                    "is_trustworthy": True,
                    "overall_score": 95.0,
                    "completeness_score": 92.0,
                    "contradiction_score": 98.0,
                    "hallucination_score": 96.0,
                    "citation_score": 94.0,
                    "reasoning": "Answer is complete and well-supported",
                },
                "metadata": {"num_chunks_retrieved": 10, "num_chunks_reranked": 5},
                "cache_hit": True,
                "cache_similarity": 0.94,
                "total_cost": 0.0,
                "latency_ms": 45,
                "timestamp": "2026-04-25T14:30:22",
            }
        }


class CacheMetrics(BaseModel):
    cache_hits: int
    cache_misses: int
    total_requests: int
    hit_rate: float
    cost_saved: float
    cache_size: int
    similarity_threshold: float

    class Config:
        json_schema_extra = {
            "example": {
                "cache_hits": 46,
                "cache_misses": 54,
                "total_requests": 100,
                "hit_rate": 46.0,
                "cost_saved": 1.75,
                "cache_size": 23,
                "similarity_threshold": 0.92,
            }
        }


class CacheInvalidateRequest(BaseModel):
    invalidate_all: bool = Field(default=False, description="Clear entire cache")
    contract_ids: list[str] | None = Field(
        default=None, description="Clear cache for specific contracts"
    )
    access_level: AccessLevel | None = Field(
        default=None, description="Clear cache for specific access level"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "invalidate_all": False,
                "contract_ids": ["cuad_0001", "cuad_0002"],
                "access_level": "admin",
            }
        }


class CacheInvalidateResponse(BaseModel):
    success: bool
    entries_cleared: int
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "entries_cleared": 12,
                "message": "Cache cleared for 2 contracts",
            }
        }


class ContractInfo(BaseModel):
    contract_id: str
    filename: str
    num_chunks: int
    confidential: bool
    created_at: datetime | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "contract_id": "cuad_0001",
                "filename": "Service_Agreement_2023.pdf",
                "num_chunks": 42,
                "confidential": True,
                "created_at": "2026-04-01T10:00:00",
            }
        }


class ContractsListResponse(BaseModel):
    contracts: list[ContractInfo]
    total_count: int
    access_level: AccessLevel

    class Config:
        json_schema_extra = {
            "example": {
                "contracts": [
                    {
                        "contract_id": "cuad_0001",
                        "filename": "Service_Agreement_2023.pdf",
                        "num_chunks": 42,
                        "confidential": True,
                    }
                ],
                "total_count": 511,
                "access_level": "admin",
            }
        }


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: dict[str, str] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2026-04-25T14:30:22",
                "services": {
                    "redis": "connected",
                    "pinecone": "connected",
                    "openai": "connected",
                    "cohere": "connected",
                },
            }
        }


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Unauthorized",
                "detail": "Invalid or expired token",
                "status_code": 401,
                "timestamp": "2026-04-25T14:30:22",
            }
        }

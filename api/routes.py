from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from api.auth import (
    TokenPayload,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_admin,
)
from api.dependencies import (
    RAGPipeline,
    check_cohere_health,
    check_openai_health,
    check_pinecone_health,
    check_redis_health,
    get_rag_pipeline,
    get_semantic_cache,
)
from api.models import (
    AccessLevel,
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    CacheMetrics,
    ContractInfo,
    ContractsListResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    VerificationResult,
)
from cache.semantic_cache import SemanticCache

router = APIRouter()


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    summary="User Login",
    description="Authenticate user and receive JWT access token",
    tags=["Authentication"],
)
async def login(request: LoginRequest) -> LoginResponse:
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        email=user["email"], access_level=user["access_level"]
    )
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        access_level=user["access_level"],
        expires_in=3600,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG Query",
    description="Ask questions about contracts using RAG pipeline",
    tags=["Query"],
)
async def query_contracts(
    request: QueryRequest,
    current_user: TokenPayload = Depends(get_current_user),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    try:
        result = await pipeline.query(
            query=request.query,
            access_level=current_user.access_level.value,
            top_k=request.top_k,
            use_cache=request.use_cache,
            rerank=request.rerank,
            verify=request.verify,
        )
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=[
                SourceChunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    contract_id=chunk.get("contract_id", ""),
                    content=chunk.get("text", ""),
                    relevance_score=chunk.get("score", 0.0),
                    rerank_score=chunk.get("rerank_score"),
                    page_number=chunk.get("page_number"),
                )
                for chunk in result.get("sources", [])
            ],
            verification=VerificationResult(
                is_trustworthy=result["verification"]["is_trustworthy"],
                overall_score=result["verification"]["overall_score"],
                completeness_score=result["verification"]["completeness_score"],
                contradiction_score=result["verification"]["contradiction_score"],
                hallucination_score=result["verification"]["hallucination_score"],
                citation_score=result["verification"]["citation_score"],
                reasoning=result["verification"].get("reasoning", ""),
            ),
            metadata=result.get("metadata", {}),
            cache_hit=result.get("cache_hit", False),
            cache_similarity=result.get("cache_similarity"),
            total_cost=result.get("total_cost", 0.0),
            latency_ms=result.get("latency_ms", 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {e!s}",
        )


@router.get(
    "/cache/metrics",
    response_model=CacheMetrics,
    summary="Cache Metrics",
    description="Get semantic cache performance metrics",
    tags=["Cache"],
)
async def get_cache_metrics(
    current_user: TokenPayload = Depends(get_current_user),
    cache: SemanticCache = Depends(get_semantic_cache),
) -> CacheMetrics:
    try:
        metrics = cache.get_metrics()
        return CacheMetrics(
            cache_hits=metrics["cache_hits"],
            cache_misses=metrics["cache_misses"],
            total_requests=metrics["total_requests"],
            hit_rate=metrics["hit_rate"],
            cost_saved=metrics["cost_saved"],
            cache_size=metrics["cache_size"],
            similarity_threshold=metrics["similarity_threshold"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache metrics: {e!s}",
        )


@router.post(
    "/cache/invalidate",
    response_model=CacheInvalidateResponse,
    summary="Invalidate Cache",
    description="Clear cache entries (admin only)",
    tags=["Cache"],
)
async def invalidate_cache(
    request: CacheInvalidateRequest,
    current_user: TokenPayload = Depends(require_admin),
    cache: SemanticCache = Depends(get_semantic_cache),
) -> CacheInvalidateResponse:
    try:
        entries_cleared = 0
        if request.invalidate_all:
            entries_cleared = cache.clear()
            message = "Successfully cleared all cache entries"
        elif request.contract_ids:
            for contract_id in request.contract_ids:
                count = cache.invalidate(contract_id=contract_id)
                entries_cleared += count
            message = f"Cleared cache for {len(request.contract_ids)} contracts"
        elif request.access_level:
            entries_cleared = cache.invalidate(access_level=request.access_level.value)
            message = f"Cleared cache for access level '{request.access_level.value}'"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must specify invalidate_all, contract_ids, or access_level",
            )
        return CacheInvalidateResponse(
            success=True, entries_cleared=entries_cleared, message=message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache invalidation failed: {e!s}",
        )


@router.get(
    "/contracts",
    response_model=ContractsListResponse,
    summary="List Contracts",
    description="Get list of available contracts (filtered by access level)",
    tags=["Contracts"],
)
async def list_contracts(
    current_user: TokenPayload = Depends(get_current_user),
) -> ContractsListResponse:
    try:
        import pandas as pd

        metadata_path = "data/cuad/metadata.csv"
        df = pd.read_csv(metadata_path)
        if current_user.access_level == AccessLevel.USER:
            df = df[df["confidential"] == False]
        contracts = [
            ContractInfo(
                contract_id=row["contract_id"],
                filename=row["filename"],
                num_chunks=row["num_chunks"],
                confidential=row["confidential"],
                created_at=None,
            )
            for _, row in df.iterrows()
        ]
        return ContractsListResponse(
            contracts=contracts,
            total_count=len(contracts),
            access_level=current_user.access_level,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Contract metadata file not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list contracts: {e!s}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and service health status",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    import asyncio

    redis_status, pinecone_status, openai_status, cohere_status = await asyncio.gather(
        check_redis_health(),
        check_pinecone_health(),
        check_openai_health(),
        check_cohere_health(),
        return_exceptions=True,
    )
    services = {
        "redis": (
            redis_status.get("status", "unhealthy")
            if isinstance(redis_status, dict)
            else "unhealthy"
        ),
        "pinecone": (
            pinecone_status.get("status", "unhealthy")
            if isinstance(pinecone_status, dict)
            else "unhealthy"
        ),
        "openai": (
            openai_status.get("status", "unhealthy")
            if isinstance(openai_status, dict)
            else "unhealthy"
        ),
        "cohere": (
            cohere_status.get("status", "unhealthy")
            if isinstance(cohere_status, dict)
            else "unhealthy"
        ),
    }
    unhealthy_count = sum((1 for status in services.values() if status == "unhealthy"))
    if unhealthy_count == 0:
        overall_status = "healthy"
    elif unhealthy_count <= 2:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        services=services,
    )


@router.get(
    "/admin/stats",
    summary="Admin Statistics",
    description="Get system statistics (admin only)",
    tags=["Admin"],
)
async def get_admin_stats(current_user: TokenPayload = Depends(require_admin)) -> dict:
    return {
        "total_queries": 10000,
        "total_cost": 205.5,
        "avg_cost_per_query": 0.021,
        "cache_hit_rate": 46.0,
        "total_cost_saved": 175.4,
        "active_users": 15,
    }

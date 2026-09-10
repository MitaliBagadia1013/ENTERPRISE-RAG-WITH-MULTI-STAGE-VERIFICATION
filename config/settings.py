from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "enterprise-contracts"
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    REDIS_URL: str = "redis://localhost:6379"
    COHERE_API_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000", "*"]
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL_SECONDS: int = 86400
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    VALID_ROLES: list[str] = [
        "admin",
        "legal",
        "finance",
        "hr",
        "procurement",
        "viewer",
    ]
    RAW_CONTRACTS_DIR: str = "data/raw_contracts"
    PARSED_CONTRACTS_DIR: str = "data/parsed_contracts"
    PROCESSED_DIR: str = "data/processed"
    CUAD_DATA_DIR: str = "data/cuad"
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    @property
    def redis_host(self) -> str:
        return self.REDIS_URL.split("://")[1].split(":")[0]

    @property
    def redis_port(self) -> int:
        url_parts = self.REDIS_URL.split(":")
        return int(url_parts[-1].split("/")[0]) if len(url_parts) > 2 else 6379

    @property
    def pinecone_metric(self) -> str:
        return "cosine"

    def validate_role(self, role: str) -> bool:
        return role in self.VALID_ROLES


settings = Settings()
assert (
    settings.EMBEDDING_DIMENSION == 3072
), "text-embedding-3-large requires 3072 dimensions"
assert (
    settings.TOP_K_RERANK <= settings.TOP_K_RETRIEVAL
), "Rerank top_k must be <= retrieval top_k"

import json
import sys
import time
from pathlib import Path
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def initialize_openai_client() -> OpenAI:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def initialize_pinecone_client() -> Pinecone:
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found")
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def create_or_get_index(pc: Pinecone, index_name: str):
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"Index '{index_name}' exists")
        return pc.Index(index_name)
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=settings.EMBEDDING_DIMENSION,
        metric=settings.pinecone_metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(10)
    return pc.Index(index_name)


def get_encoding():
    try:
        return tiktoken.encoding_for_model("text-embedding-3-large")
    except:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    encoding = get_encoding()
    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int = 7900) -> str:
    encoding = get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens) + "[TRUNCATED]"


def validate_and_fix_batch(texts: list[str]) -> list[str]:
    MAX_TOKENS = 7900
    fixed_texts = []
    truncated_count = 0
    for text in texts:
        token_count = count_tokens(text)
        if token_count > MAX_TOKENS:
            truncated_count += 1
            fixed_text = truncate_to_tokens(text, MAX_TOKENS)
            fixed_texts.append(fixed_text)
        else:
            fixed_texts.append(text)
    if truncated_count > 0:
        print(f"Truncated {truncated_count}/{len(texts)} texts (tiktoken)")
    return fixed_texts


def generate_embeddings(
    client: OpenAI, texts: list[str], model: str = "text-embedding-3-large"
) -> list[list[float]]:
    texts = validate_and_fix_batch(texts)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str and attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"\nRate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def prepare_vectors(
    chunks: list[dict], embeddings: list[list[float]]
) -> list[tuple[str, list[float], dict]]:
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = {
            "text": chunk["text"][:1000],
            "contract_id": chunk["contract_id"],
            "rbac_roles": chunk["rbac_roles"],
            "contract_type": chunk["contract_type"],
            "word_count": chunk["word_count"],
            "chunk_index": chunk["chunk_index"],
        }
        vectors.append((chunk["chunk_id"], embedding, metadata))
    return vectors


def upload_to_pinecone(
    index, vectors: list[tuple[str, list[float], dict]], batch_size: int = 100
):
    total = len(vectors)
    print(f"\nUploading {total:,} vectors...")
    for i in tqdm(range(0, total, batch_size), desc="Uploading"):
        batch = vectors[i : i + batch_size]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                index.upsert(vectors=batch)
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise
    print(f"Uploaded {total:,} vectors!")


def embed_and_index_chunks():
    print("=" * 70)
    print("Embedder - WITH TIKTOKEN VALIDATION")
    print("=" * 70)
    print("\nInitializing...")
    openai_client = initialize_openai_client()
    pinecone_client = initialize_pinecone_client()
    print("Clients ready")
    chunks_file = Path(settings.CUAD_DATA_DIR) / "chunks.json"
    print("\nLoading chunks...")
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks):,} chunks")
    print("\nSetting up Pinecone...")
    index = create_or_get_index(pinecone_client, settings.PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    existing = stats.total_vector_count
    print(f"Current vectors: {existing:,}")
    if existing > 0:
        print("\nSMART RESUME")
        print(f"Skipping {existing:,} already uploaded")
        chunks = chunks[existing:]
        print(f"Remaining: {len(chunks):,}")
        print(f"Savings: ${existing * 0.00013:.2f}")
    print("\nGenerating embeddings...")
    print(f"Chunks: {len(chunks):,}")
    print(f"Cost: ${len(chunks) * 0.00013:.2f}")
    texts = [chunk["text"] for chunk in chunks]
    BATCH_SIZE = 100
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_embeddings = generate_embeddings(
            openai_client, batch_texts, settings.EMBEDDING_MODEL
        )
        all_embeddings.extend(batch_embeddings)
        time.sleep(0.1)
    print(f"\nGenerated {len(all_embeddings):,} embeddings")
    print("\nPreparing vectors...")
    vectors = prepare_vectors(chunks, all_embeddings)
    print(f"{len(vectors):,} vectors ready")
    upload_to_pinecone(index, vectors)
    print("\nVerifying...")
    time.sleep(2)
    final_stats = index.describe_index_stats()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated: {len(all_embeddings):,} embeddings")
    print(f"Uploaded: {len(vectors):,} vectors")
    print(f"Total: {final_stats.total_vector_count:,} in Pinecone")
    print(f"Cost: ${len(chunks) * 0.00013:.2f}")
    print("\nComplete!")


if __name__ == "__main__":
    embed_and_index_chunks()

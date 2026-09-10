import json
import re
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def split_into_sentences(text: str) -> list[str]:
    abbreviations = [
        "\\bInc\\.",
        "\\bCorp\\.",
        "\\bLtd\\.",
        "\\bLLC\\.",
        "\\bU\\.S\\.",
        "\\bMr\\.",
        "\\bMrs\\.",
        "\\bDr\\.",
        "\\bPh\\.D\\.",
        "\\bNo\\.",
        "\\bvol\\.",
        "\\bFig\\.",
        "\\bEq\\.",
        "\\bSec\\.",
        "\\bArt\\.",
        "\\bEt\\s+al\\.",
        "\\be\\.g\\.",
        "\\bi\\.e\\.",
        "\\bet\\s+al\\.",
    ]
    for abbr in abbreviations:
        text = re.sub(
            abbr,
            lambda m: m.group(0).replace(".", "<PERIOD>"),
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub("(\\d+)\\.(\\d+)", "\\1<PERIOD>\\2", text)
    text = re.sub("(\\d+)\\.(\\d+)", "\\1<PERIOD>\\2", text)
    sentences = re.split("(?<=[.!?])\\s+(?=[A-Z\\n])", text)
    sentences = [s.replace("<PERIOD>", ".") for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def count_words(text: str) -> int:
    return len(text.split())


def create_chunks(
    text: str, chunk_size_words: int = 512, overlap_words: int = 50
) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    for sentence in sentences:
        sentence_word_count = count_words(sentence)
        if (
            current_word_count + sentence_word_count > chunk_size_words
            and current_chunk_sentences
        ):
            chunk_text = "".join(current_chunk_sentences)
            chunks.append(chunk_text)
            overlap_sentences = []
            overlap_word_count = 0
            for sent in reversed(current_chunk_sentences):
                sent_words = count_words(sent)
                if overlap_word_count + sent_words <= overlap_words:
                    overlap_sentences.insert(0, sent)
                    overlap_word_count += sent_words
                else:
                    break
            current_chunk_sentences = overlap_sentences
            current_word_count = overlap_word_count
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_word_count
    if current_chunk_sentences:
        chunk_text = "".join(current_chunk_sentences)
        chunks.append(chunk_text)
    return chunks


def load_contract_metadata() -> pd.DataFrame:
    metadata_path = Path(settings.CUAD_DATA_DIR) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def create_chunk_metadata(
    contract_id: str, chunk_idx: int, chunk_text: str, contract_metadata: dict
) -> dict:
    return {
        "chunk_id": f"{contract_id}_chunk_{chunk_idx}",
        "contract_id": contract_id,
        "chunk_index": chunk_idx,
        "text": chunk_text,
        "word_count": count_words(chunk_text),
        "char_count": len(chunk_text),
        "rbac_roles": contract_metadata.get("rbac_roles", "admin,viewer").split(","),
        "contract_type": contract_metadata.get("contract_type", "Unknown"),
        "filename": contract_metadata.get("filename", ""),
    }


def process_all_contracts():
    print("=" * 70)
    print("Chunker - Sentence-Aware Chunking with RBAC Metadata")
    print("=" * 70)
    processed_dir = Path(settings.CUAD_DATA_DIR) / "processed"
    output_file = Path(settings.CUAD_DATA_DIR) / "chunks.json"
    print(f"\nInput: {processed_dir}")
    print(f"Output: {output_file}")
    print("\nLoading contract metadata...")
    try:
        metadata_df = load_contract_metadata()
        print(f"Loaded metadata for {len(metadata_df)} contracts")
    except Exception as e:
        print(f"\nError loading metadata: {e}")
        print("Run 'python -m data.load_cuad' first to generate metadata.csv")
        return
    contract_files = sorted(processed_dir.glob("cuad_*.txt"))
    if not contract_files:
        print("\nNo processed contract files found!")
        print("Run 'python -m ingestion.document_parser' first.")
        return
    print(f"\nProcessing {len(contract_files)} contracts...")
    print(f"Chunk size: {settings.CHUNK_SIZE} words")
    print(f"Overlap: {settings.CHUNK_OVERLAP} words\n")
    all_chunks = []
    total_chunks = 0
    total_words = 0
    rbac_distribution = {}
    contract_type_distribution = {}
    failed_contracts = []
    for contract_file in tqdm(contract_files, desc="Chunking contracts"):
        try:
            contract_id = contract_file.stem
            with open(contract_file, "r", encoding="utf-8") as f:
                contract_text = f.read()
            contract_meta_row = metadata_df[metadata_df["contract_id"] == contract_id]
            if contract_meta_row.empty:
                print(
                    f"\nWarning: No metadata found for {contract_id}, using defaults"
                )
                contract_meta = {
                    "rbac_roles": "admin,viewer",
                    "contract_type": "Unknown",
                    "filename": contract_file.name,
                }
            else:
                contract_meta = contract_meta_row.iloc[0].to_dict()
            chunks = create_chunks(
                contract_text,
                chunk_size_words=settings.CHUNK_SIZE,
                overlap_words=settings.CHUNK_OVERLAP,
            )
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_metadata = create_chunk_metadata(
                    contract_id, chunk_idx, chunk_text, contract_meta
                )
                all_chunks.append(chunk_metadata)
                total_chunks += 1
                total_words += chunk_metadata["word_count"]
                roles_key = ",".join(sorted(chunk_metadata["rbac_roles"]))
                rbac_distribution[roles_key] = rbac_distribution.get(roles_key, 0) + 1
                contract_type = chunk_metadata["contract_type"]
                contract_type_distribution[contract_type] = (
                    contract_type_distribution.get(contract_type, 0) + 1
                )
        except Exception as e:
            failed_contracts.append((contract_file.name, str(e)))
            print(f"\nError processing {contract_file.name}: {e}")
    print(f"\nSaving {len(all_chunks)} chunks to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    file_size_mb = output_file.stat().st_size / 1000000
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = len(contract_files) - len(failed_contracts)
    print(f"Successfully processed: {successful}/{len(contract_files)} contracts")
    if failed_contracts:
        print(f"\nFailed contracts: {len(failed_contracts)}")
        for name, error in failed_contracts[:5]:
            print(f"- {name}: {error}")
    print("\nChunk Statistics:")
    print(f"Total chunks: {total_chunks:,}")
    print(f"Avg chunks/contract: {total_chunks / successful:.1f}")
    print(f"Avg words/chunk: {total_words / total_chunks:.1f}")
    print(f"Output file size: {file_size_mb:.2f} MB")
    print("\nRBAC Distribution (top 5):")
    sorted_rbac = sorted(rbac_distribution.items(), key=lambda x: x[1], reverse=True)
    for roles, count in sorted_rbac[:5]:
        pct = 100 * count / total_chunks
        print(f"{roles:30s} {count:6,} chunks ({pct:5.1f}%)")
    print("\nContract Type Distribution (top 10):")
    sorted_types = sorted(
        contract_type_distribution.items(), key=lambda x: x[1], reverse=True
    )
    for contract_type, count in sorted_types[:10]:
        pct = 100 * count / total_chunks
        print(f"{contract_type:30s} {count:6,} chunks ({pct:5.1f}%)")
    print("\nNext Steps:")
    print("1. Review chunk statistics above")
    print("2. Run: python -m ingestion.embedder")
    print("3. Then: python -m retrieval.pinecone_retriever")
    print("\nChunking complete!")


if __name__ == "__main__":
    process_all_contracts()

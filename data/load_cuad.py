import json
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def assign_rbac_roles(contract_text: str, contract_title: str = "") -> List[str]:
    text_to_analyze = (contract_title + "" + contract_text[:2000]).lower()
    employment_keywords = [
        "employment agreement",
        "employee",
        "compensation",
        "salary",
        "severance",
        "termination of employment",
        "executive",
        "benefits",
        "bonus",
        "stock option",
    ]
    if any((keyword in text_to_analyze for keyword in employment_keywords)):
        return ["admin", "legal", "hr"]
    financial_keywords = [
        "credit agreement",
        "loan",
        "financing",
        "interest rate",
        "promissory note",
        "debt",
        "lender",
        "borrower",
        "payment terms",
        "installment",
        "principal amount",
    ]
    if any((keyword in text_to_analyze for keyword in financial_keywords)):
        return ["admin", "finance"]
    ma_keywords = [
        "merger",
        "acquisition",
        "purchase price",
        "closing date",
        "due diligence",
        "representations and warranties",
        "stock purchase",
        "asset purchase",
    ]
    if any((keyword in text_to_analyze for keyword in ma_keywords)):
        return ["admin", "legal", "finance"]
    nda_keywords = [
        "non-disclosure",
        "confidential information",
        "proprietary",
        "trade secret",
        "nda",
        "confidentiality agreement",
    ]
    if any((keyword in text_to_analyze for keyword in nda_keywords)):
        return ["admin", "legal"]
    service_keywords = [
        "master service",
        "statement of work",
        "deliverables",
        "service provider",
        "vendor",
        "supplier",
        "professional services",
        "consulting agreement",
    ]
    if any((keyword in text_to_analyze for keyword in service_keywords)):
        return ["admin", "procurement", "legal"]
    lease_keywords = [
        "lease agreement",
        "lessor",
        "lessee",
        "premises",
        "rent",
        "landlord",
        "tenant",
        "property lease",
    ]
    if any((keyword in text_to_analyze for keyword in lease_keywords)):
        return ["admin", "legal", "finance"]
    ip_keywords = [
        "intellectual property",
        "patent",
        "copyright",
        "trademark",
        "license agreement",
        "royalty",
        "licensing",
        "ip rights",
    ]
    if any((keyword in text_to_analyze for keyword in ip_keywords)):
        return ["admin", "legal"]
    return ["admin", "viewer"]


def load_cuad_dataset():
    print("Downloading CUAD dataset from HuggingFace...")
    print("(This will download ~500MB of contract data)")
    dataset = load_dataset("TheAtticusProject/cuad")
    print(f"Loaded CUAD dataset")
    print(f"Train examples: {len(dataset['train'])}")
    return dataset


def process_cuad_contracts(dataset):
    contracts_dir = Path(settings.CUAD_DATA_DIR) / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {contracts_dir}")
    print("Extracting text from PDFs and assigning RBAC roles...\n")
    metadata_records = []
    for idx, record in enumerate(tqdm(dataset["train"], desc="Processing contracts")):
        try:
            pdf = record.get("pdf")
            if pdf is None:
                continue
            contract_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    contract_text += page_text + "\n"
            if len(contract_text) < 100:
                print(
                    f"Skipping contract {idx} - too short ({len(contract_text)} chars)"
                )
                continue
            title = f"Contract_{idx:04d}"
            allowed_roles = assign_rbac_roles(contract_text, title)
            contract_id = f"cuad_{idx:04d}"
            filename = f"{contract_id}.txt"
            filepath = contracts_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(contract_text)
            metadata_records.append(
                {
                    "contract_id": contract_id,
                    "filename": filename,
                    "title": title,
                    "allowed_roles": ",".join(allowed_roles),
                    "source": "CUAD",
                    "char_count": len(contract_text),
                    "word_count": len(contract_text.split()),
                }
            )
        except Exception as e:
            print(f"Error processing contract {idx}: {str(e)}")
            continue
    return metadata_records


def extract_qa_pairs(dataset):
    print("\nCreating evaluation Q&A pairs from contracts...")
    qa_pairs = []
    print(
        "Note: Using placeholder Q&A pairs. Load CUAD Q&A dataset separately for full evaluation."
    )
    qa_pairs.append(
        {
            "contract_id": "cuad_placeholder",
            "question": "What are the key terms of this contract?",
            "answer": "See contract for full terms and conditions.",
            "context": "Placeholder - will be replaced with actual CUAD Q&A pairs",
        }
    )
    print(f"Created {len(qa_pairs)} placeholder Q&A pairs")
    return qa_pairs


def main():
    print("=" * 70)
    print("CUAD Dataset Loader with RBAC Assignment")
    print("=" * 70)
    dataset = load_cuad_dataset()
    metadata_records = process_cuad_contracts(dataset)
    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = Path(settings.CUAD_DATA_DIR) / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata to: {metadata_path}")
    print("\nRBAC Role Distribution:")
    print(metadata_df["allowed_roles"].value_counts())
    qa_pairs = extract_qa_pairs(dataset)
    qa_path = Path(settings.CUAD_DATA_DIR) / "eval_dataset.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved Q&A pairs to: {qa_path}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Contracts processed: {len(metadata_records)}")
    print(f"Q&A pairs extracted: {len(qa_pairs)}")
    print(f"Average contract length: {metadata_df['word_count'].mean():.0f} words")
    print(f"Total data size: {metadata_df['char_count'].sum() / 1000000:.1f} MB")
    print("\nNext Steps:")
    print("1. Run: python -m ingestion.document_parser")
    print("2. Then: python -m ingestion.chunker")
    print("3. Then: python -m ingestion.embedder")
    print("\nCUAD dataset ready for ingestion!")


if __name__ == "__main__":
    main()

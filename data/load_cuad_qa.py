import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings

CUAD_CATEGORIES = [
    "cuad_affiliate_license-licensee",
    "cuad_affiliate_license-licensor",
    "cuad_anti-assignment",
    "cuad_audit_rights",
    "cuad_cap_on_liability",
    "cuad_change_of_control",
    "cuad_competitive_restriction_exception",
    "cuad_covenant_not_to_sue",
    "cuad_effective_date",
    "cuad_exclusivity",
    "cuad_expiration_date",
    "cuad_governing_law",
    "cuad_insurance",
    "cuad_ip_ownership_assignment",
    "cuad_irrevocable_or_perpetual_license",
    "cuad_joint_ip_ownership",
    "cuad_license_grant",
    "cuad_liquidated_damages",
    "cuad_minimum_commitment",
    "cuad_most_favored_nation",
    "cuad_no-solicit_of_customers",
    "cuad_no-solicit_of_employees",
    "cuad_non-compete",
    "cuad_non-disparagement",
    "cuad_non-transferable_license",
    "cuad_notice_period_to_terminate_renewal",
    "cuad_post-termination_services",
    "cuad_price_restrictions",
    "cuad_renewal_term",
    "cuad_revenue-profit_sharing",
    "cuad_rofr-rofo-rofn",
    "cuad_source_code_escrow",
    "cuad_termination_for_convenience",
    "cuad_third_party_beneficiary",
    "cuad_uncapped_liability",
    "cuad_unlimited-all-you-can-eat-license",
    "cuad_volume_restriction",
    "cuad_warranty_duration",
]


def extract_cuad_qa_pairs():
    print("=" * 70)
    print("CUAD Q&A Extractor")
    print("=" * 70)
    print(
        f"\nExtracting Q&A pairs from {len(CUAD_CATEGORIES)} legal clause categories..."
    )
    print("(This will download ~100MB from HuggingFace)")
    all_qa_pairs = []
    category_counts = {}
    for category in tqdm(CUAD_CATEGORIES, desc="Loading categories"):
        try:
            dataset = load_dataset("nguha/legalbench", category, trust_remote_code=True)
            if "train" in dataset:
                for idx, record in enumerate(dataset["train"]):
                    qa_pair = {
                        "contract_id": f"cuad_{idx:04d}",
                        "category": category.replace("cuad_", "")
                        .replace("-", "_")
                        .replace("_", "")
                        .title(),
                        "question": record.get("text", ""),
                        "context": record.get("text", "")[:1000],
                        "answer": record.get("answer", ""),
                        "label": record.get("label", None),
                    }
                    all_qa_pairs.append(qa_pair)
                category_counts[category] = len(dataset["train"])
            if "test" in dataset:
                for idx, record in enumerate(dataset["test"]):
                    qa_pair = {
                        "contract_id": f"cuad_test_{idx:04d}",
                        "category": category.replace("cuad_", "")
                        .replace("-", "_")
                        .replace("_", "")
                        .title(),
                        "question": record.get("text", ""),
                        "context": record.get("text", "")[:1000],
                        "answer": record.get("answer", ""),
                        "label": record.get("label", None),
                    }
                    all_qa_pairs.append(qa_pair)
                category_counts[category] = category_counts.get(category, 0) + len(
                    dataset["test"]
                )
        except Exception as e:
            print(f"\nError loading {category}: {str(e)}")
            continue
    return (all_qa_pairs, category_counts)


def main():
    qa_pairs, category_counts = extract_cuad_qa_pairs()
    output_path = Path(settings.CUAD_DATA_DIR) / "qa_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Q&A pairs extracted: {len(qa_pairs)}")
    print(f"Categories processed: {len(category_counts)}")
    print(f"Saved to: {output_path}")
    print("\nTop 10 Categories by Q&A Count:")
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )
    for category, count in sorted_categories[:10]:
        print(f"{category}: {count} Q&A pairs")
    print("\nNext Steps:")
    print("Use this Q&A dataset for RAGAS evaluation")
    print("Each Q&A pair tests retrieval accuracy on specific legal clauses")
    print("\nCUAD Q&A dataset ready for evaluation!")


if __name__ == "__main__":
    main()

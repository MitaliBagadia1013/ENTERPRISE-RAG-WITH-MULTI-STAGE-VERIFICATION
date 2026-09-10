import re
import sys
import unicodedata
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def remove_sgml_tags(text: str) -> str:
    text = re.sub("<[^>]+>", "", text)
    text = re.sub("<|>", "", text)
    return text


def remove_boilerplate(text: str) -> str:
    text = re.sub("EXHIBIT\\s+\\d+\\.?\\d*", "", text, flags=re.IGNORECASE)
    text = re.sub("Page\\s+\\d+(\\s+of\\s+\\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub("-\\s*\\d+\\s*-", "", text)
    text = re.sub("Form\\s+(10-K|10-Q|8-K|S-1)", "", text, flags=re.IGNORECASE)
    text = re.sub("File\\s+No\\.?\\s+\\d+-\\d+", "", text, flags=re.IGNORECASE)
    text = re.sub("\\(continued\\)", "", text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    text = text.replace("\t", "")
    text = re.sub("{2,}", "", text)
    text = re.sub("^ +| +$", "", text, flags=re.MULTILINE)
    text = re.sub("\\n{3,}", "\n\n", text)
    return text


def handle_non_ascii(text: str) -> str:
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(', "\'").replace(', "'")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("•", "*").replace("·", "*")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        (char if ord(char) < 128 or char in "©®§°" else "" for char in text)
    )
    return text


def standardize_line_breaks(text: str) -> str:
    text = re.sub("-\\s*\\n\\s*", "-", text)
    text = re.sub("([a-z])\\n([a-z])", "\\1 \\2", text)
    text = re.sub("([.!?])([A-Z])", "\\1 \\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def remove_empty_sections(text: str) -> str:
    lines = text.split("\n")
    meaningful_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and (len(stripped) > 2 or stripped.isdigit()):
            meaningful_lines.append(line)
    text = "\n".join(meaningful_lines)
    return text


def validate_cleaned_text(text: str, min_length: int = 500) -> tuple[bool, str]:
    if len(text) < min_length:
        return (
            False,
            f"Text too short after cleaning: {len(text)} chars (min: {min_length})",
        )
    if not any((c.isalpha() for c in text)):
        return (False, "No alphabetic characters found")
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum((1 for c in alpha_chars if c.isupper())) / len(alpha_chars)
        if upper_ratio > 0.9:
            return (
                False,
                f"Text is {upper_ratio * 100:.1f}% uppercase (possible OCR issue)",
            )
    return (True, "")


def clean_contract_text(text: str) -> tuple[str, list[str]]:
    warnings = []
    original_length = len(text)
    text = remove_sgml_tags(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    text = handle_non_ascii(text)
    text = standardize_line_breaks(text)
    text = remove_empty_sections(text)
    text = text.strip()
    is_valid, error_msg = validate_cleaned_text(text)
    if not is_valid:
        warnings.append(f"Validation warning: {error_msg}")
    reduction = 100 * (1 - len(text) / original_length)
    if reduction > 50:
        warnings.append(f"Significant size reduction: {reduction:.1f}%")
    return (text, warnings)


def process_all_contracts():
    print("=" * 70)
    print("Document Parser - 7-Stage Contract Cleaning Pipeline")
    print("=" * 70)
    input_dir = Path(settings.CUAD_DATA_DIR) / "contracts"
    output_dir = Path(settings.CUAD_DATA_DIR) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    contract_files = sorted(input_dir.glob("cuad_*.txt"))
    if not contract_files:
        print("\nNo contract files found! Run load_cuad.py first.")
        return
    print(f"\nProcessing {len(contract_files)} contracts...\n")
    total_warnings = 0
    total_original_size = 0
    total_cleaned_size = 0
    failed_contracts = []
    for contract_file in tqdm(contract_files, desc="Cleaning contracts"):
        try:
            with open(contract_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
            total_original_size += len(raw_text)
            cleaned_text, warnings = clean_contract_text(raw_text)
            total_cleaned_size += len(cleaned_text)
            total_warnings += len(warnings)
            if warnings:
                print(f"\n{contract_file.name}:")
                for warning in warnings:
                    print(f"- {warning}")
            output_file = output_dir / contract_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
        except Exception as e:
            failed_contracts.append((contract_file.name, str(e)))
            print(f"\nError processing {contract_file.name}: {e}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = len(contract_files) - len(failed_contracts)
    print(f"Successfully processed: {successful}/{len(contract_files)} contracts")
    if failed_contracts:
        print(f"\nFailed contracts: {len(failed_contracts)}")
        for name, error in failed_contracts[:5]:
            print(f"- {name}: {error}")
    reduction_pct = 100 * (1 - total_cleaned_size / total_original_size)
    print("\nSize Statistics:")
    print(f"Original size: {total_original_size / 1000000:.2f} MB")
    print(f"Cleaned size: {total_cleaned_size / 1000000:.2f} MB")
    print(f"Reduction: {reduction_pct:.1f}%")
    print(f"\nTotal warnings: {total_warnings}")
    print("\nNext Steps:")
    print("1. Review warnings above (if any)")
    print("2. Run: python -m ingestion.chunker")
    print("3. Then: python -m ingestion.embedder")
    print("\nDocument parsing complete!")


if __name__ == "__main__":
    process_all_contracts()

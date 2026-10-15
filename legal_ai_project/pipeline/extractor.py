import re
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python pipeline/extractor.py <case_file>")
    exit()

case_name = sys.argv[1]

INPUT_FILE = f"data/raw/{case_name}.txt"
OUTPUT_FILE = f"data/metadata/{case_name}.json"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

metadata = {
    "case_title": None,
    "court": None,
    "judgment_date": None,
    "case_number": None
}

title_match = re.search(
    r"(.+?)\s+vs\s+(.+?)\s+on\s+(\d{1,2}\s+\w+,\s+\d{4})",
    text,
    re.IGNORECASE
)

if title_match:
    metadata["case_title"] = (
        title_match.group(1).strip()
        + " vs "
        + title_match.group(2).strip()
    )
    metadata["judgment_date"] = title_match.group(3).strip()

if "SUPREME COURT OF INDIA" in text.upper():
    metadata["court"] = "Supreme Court of India"

case_number_match = re.search(
    r"^\s*(CRIMINAL\s+APPEAL\s+NOS?\.\s*[\d,\-\s]+OF\s+\d{4})\s*$",
    text,
    re.IGNORECASE | re.MULTILINE
)

if case_number_match:
    metadata["case_number"] = case_number_match.group(1).strip()

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

print(f"Metadata extracted: {case_name}")
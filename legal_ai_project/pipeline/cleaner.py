"""
Document Cleaning Agent for Indian Kanoon Judgments.
Removes website navigation, menus, advertisements, AI tags, and web boilerplate
while strictly preserving facts, legal reasoning, provisions, findings, and operative orders.
"""

import argparse
import html
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    RAW_DIR,
    PROCESSED_DIR,
    PIPELINE_LOG_FILE
)

logger = logging.getLogger(__name__)

# Exact or substring phrases to remove (navigation, UI, promos)
UI_BOILERPLATE_PATTERNS = [
    r"Skip\s+to\s+main\s+content",
    r"Indian\s+Kanoon\s+-\s+Search\s+engine\s+for\s+Indian\s+Law",
    r"Search\s+Indian\s+laws\s+and\s+judgments",
    r"Main\s+Navigation",
    r"Mobile\s+Navigation",
    r"Free\s+features",
    r"Premium\s+features",
    r"Prism\s+AI",
    r"IKademy",
    r"Legal\s+Document\s+View",
    r"Tools\s+for\s+analyzing\s+structure\s+and\s+cite\s+text\s+of\s+judgments",
    r"Unlock\s+Advanced\s+Research\s+with\s+PRISM\s+AI",
    r"Unlock\s+Advanced\s+Research\s+with",
    r"Integrated\s+with\s+over\s+\d+\s+crore\s+judgments\s+and\s+laws[^\n]*",
    r"Know\s+your\s+Kanoon",
    r"Doc\s+Gen\s+Hub",
    r"Counter\s+Argument",
    r"Case\s+Predict\s+AI",
    r"Talk\s+with\s+IK\s+Doc",
    r"Upgrade\s+to\s+Premium",
    r"Document\s+Options",
    r"Get\s+in\s+PDF",
    r"Print\s+it!?",
    r"Download\s+Court\s+Copy",
    r"Case\s+Recast\s+AI",
    r"\[Cites\s+\d+\s*,\s*Cited\s+by\s+\d+\]",
    r"\[Cites\s+\d+\s*\]",
    r"\[Cited\s+by\s+\d+\]"
]

# Footer boilerplate start markers
FOOTER_START_MARKERS = [
    r"Related\s+AI\s+tags,\s+queries\s+and\s+research\s+notes",
    r"Top\s+AI\s+Tags",
    r"Related\s+user\s+Queries",
    r"About\s*\n\s*Disclaimer\s*\n\s*Privacy\s+Policy",
    r"Case\s+Removal\s*\n\s*Blog\s*\n\s*Share\s+URL",
    r"\n\s*About\s+Disclaimer\s+Privacy\s+Policy\s+Terms",
    r"Take\s+notes\s+as\s+you\s+read\s+a\s+judgment"
]


class DocumentCleaner:
    """Cleans raw judgment text by stripping web UI noise while preserving legal judgment content."""

    def __init__(self):
        pass

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters, quotes, dashes, and unescape HTML entities."""
        # Unescape HTML entities (&nbsp;, &amp;, &lt;, etc.)
        text = html.unescape(text)

        # Standardize unicode quotes and hyphens
        replacements = {
            "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
            "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
            "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
            "\u00a0": " ", "\u200b": "", "\ufeff": "", "\t": " "
        }
        for orig, rep in replacements.items():
            text = text.replace(orig, rep)

        text = unicodedata.normalize("NFKD", text)
        return text

    def remove_header_navigation(self, text: str) -> str:
        """Remove top Indian Kanoon navigation bar and promo elements."""
        lines = text.split("\n")
        cleaned_lines = []
        in_header_skip = False

        # Identify header area (first 60 lines max)
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip standalone UI bullet lines
            if stripped in ("-", "...", "•", "Pricing", "Login", "Search", "PRISM", "AI"):
                if i < 60:
                    continue

            # Check for known UI patterns
            is_ui_line = False
            for pattern in UI_BOILERPLATE_PATTERNS:
                if re.search(pattern, stripped, re.IGNORECASE):
                    is_ui_line = True
                    break

            if is_ui_line and i < 80:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def remove_footer_boilerplate(self, text: str) -> str:
        """Remove bottom AI tags, user queries, and website disclaimer links."""
        cleaned_text = text

        for marker_pattern in FOOTER_START_MARKERS:
            match = re.search(marker_pattern, cleaned_text, re.IGNORECASE)
            if match:
                # Truncate anything after the footer marker
                cleaned_text = cleaned_text[:match.start()].strip()

        # Remove trailing single word menu lines at the very end
        end_lines = cleaned_text.split("\n")
        while end_lines and end_lines[-1].strip().lower() in (
            "about", "disclaimer", "privacy policy", "terms", "case removal",
            "blog", "share url", "mobile view", "case recast ai", "translation",
            "top ai tags", "law-regarding-motor-vehicles", ""
        ):
            end_lines.pop()

        return "\n".join(end_lines)

    def remove_inline_artifacts(self, text: str) -> str:
        """Remove remaining HTML tags, IK doc fragments, and orphan link lines."""
        # Strip any leftover HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove repeated Indian Kanoon link artifact tags like (http://...)
        text = re.sub(r"https?://(?:www\.)?indiankanoon\.org/[^\s\)\"\']+", "", text)

        # Normalize multiple spaces per line
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize multiple consecutive blank lines to at most two
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        return text.strip()

    def clean(self, raw_text: str) -> str:
        """Execute full cleaning pipeline on raw judgment text."""
        if not raw_text:
            return ""

        # Step 1: Unicode & HTML entity normalization
        text = self.normalize_unicode(raw_text)

        # Step 2: Remove header navigation and UI promos
        text = self.remove_header_navigation(text)

        # Step 3: Remove footer boilerplate, tags, and disclaimers
        text = self.remove_footer_boilerplate(text)

        # Step 4: Inline artifacts and whitespace normalization
        text = self.remove_inline_artifacts(text)

        return text

    def clean_and_save(self, case_id: str) -> Optional[str]:
        """Clean a single case file from data/raw/ and save to data/processed/."""
        raw_file = RAW_DIR / f"{case_id}.txt"
        if not raw_file.exists():
            clean_name = case_id.replace("case_", "")
            alt_file = RAW_DIR / f"{clean_name}.txt"
            if alt_file.exists():
                raw_file = alt_file
            else:
                logger.error("Raw file not found: %s", raw_file)
                return None

        try:
            raw_text = raw_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error("Failed to read raw file %s: %s", raw_file, e)
            return None

        cleaned_text = self.clean(raw_text)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed_file = PROCESSED_DIR / f"{case_id}.txt"

        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        logger.info("Cleaned %s: %d raw chars -> %d clean chars (saved to %s)",
                    case_id, len(raw_text), len(cleaned_text), processed_file.name)
        return cleaned_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Clean raw legal judgments")
    parser.add_argument("case_file", nargs="?", help="Specific case ID/file to clean (optional)")
    args = parser.parse_args()

    cleaner = DocumentCleaner()

    if args.case_file:
        clean_id = args.case_file.replace(".txt", "").replace(".json", "")
        if not clean_id.startswith("case_") and clean_id.isdigit():
            clean_id = f"case_{clean_id}"
        cleaner.clean_and_save(clean_id)
    else:
        raw_files = list(RAW_DIR.glob("*.txt"))
        logger.info("Cleaning %d raw judgment files...", len(raw_files))
        for rf in raw_files:
            cleaner.clean_and_save(rf.stem)
        logger.info("All documents cleaned successfully.")
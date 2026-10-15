"""
Semantic Similar-Case Retrieval Engine for Legal Precedent Analysis.
Uses FAISS IndexFlatL2 to search validated historical judgments based on
semantic meaning rather than exact keyword matches, retrieving historical outcomes as evidence.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    DATABASE_DIR,
    METADATA_DIR,
    PREDICTION_DATASET_CSV
)

logger = logging.getLogger(__name__)

INDEX_FILE = DATABASE_DIR / "legal_cases.index"
CASE_NAMES_FILE = DATABASE_DIR / "case_names.txt"
CASE_MAPPING_FILE = DATABASE_DIR / "case_mapping.json"


class LegalCaseRetriever:
    """Semantic retrieval engine querying FAISS IndexFlatL2 on validated legal precedents."""

    def __init__(
        self,
        index_file: Path = INDEX_FILE,
        case_mapping_file: Path = CASE_MAPPING_FILE,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_file = Path(index_file)
        self.case_mapping_file = Path(case_mapping_file)
        self.model_name = model_name
        self._model = None
        self._index = None
        self._mapping = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def index(self) -> faiss.Index:
        """Load and cache FAISS index."""
        if self._index is None:
            if not self.index_file.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {self.index_file}. "
                    f"Please run 'python database/vector_store.py' first."
                )
            self._index = faiss.read_index(str(self.index_file))
        return self._index

    @property
    def mapping(self) -> List[Dict[str, Any]]:
        """Load and cache case mapping metadata registry."""
        if self._mapping is None:
            if not self.case_mapping_file.exists():
                raise FileNotFoundError(
                    f"Case mapping registry not found at {self.case_mapping_file}. "
                    f"Please run 'python database/vector_store.py' first."
                )
            with open(self.case_mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._mapping = data.get("cases", [])
        return self._mapping

    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic search for Top-K most similar historical judgments.
        Returns a list of structured case dictionaries sorted by L2 distance (ascending).
        """
        if not query_text or not query_text.strip():
            return []

        # Generate 384-dimensional query vector
        query_vector = self.model.encode([query_text], convert_to_numpy=True).astype("float32")

        # Clamp top_k to total indexed cases
        available_cases = self.index.ntotal
        top_k = min(max(1, top_k), available_cases)

        # Search FAISS IndexFlatL2
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (dist, idx_pos) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx_pos < 0 or idx_pos >= len(self.mapping):
                continue

            case_info = dict(self.mapping[idx_pos])
            case_info["rank"] = rank
            case_info["distance"] = round(float(dist), 4)
            results.append(case_info)

        return results

    @staticmethod
    def format_results_as_text(results: List[Dict[str, Any]], query_text: str) -> str:
        """Format retrieval results into clear, human-readable evidence cards."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("                 SEMANTIC SIMILAR-CASE RETRIEVAL")
        lines.append("                 (FAISS IndexFlatL2 - Euclidean Distance)")
        lines.append("=" * 70)
        lines.append(f"NEW CASE QUERY:")
        lines.append(f"\"{query_text}\"")
        lines.append("=" * 70)

        if not results:
            lines.append("\n[INFO] No matching historical cases found.")
            lines.append("=" * 70)
            return "\n".join(lines)

        lines.append(f"\nRetrieved {len(results)} relevant historical precedent(s) as evidence:\n")

        for r in results:
            lines.append(f"--- [Rank #{r['rank']}] L2 Distance: {r['distance']:.4f} (Lower = More Similar) ---")
            lines.append(f"  Case ID          : {r.get('case_id')}")
            lines.append(f"  Title            : {r.get('case_title')}")
            lines.append(f"  Court            : {r.get('court')}")
            lines.append(f"  Judgment Date    : {r.get('judgment_date')}")
            lines.append(f"  Case Number      : {r.get('case_number') or 'N/A'}")

            # Parties
            parties = r.get("parties", {})
            if parties:
                if parties.get("appellant_petitioner"):
                    lines.append(f"  Appellant/Claim. : {parties.get('appellant_petitioner')}")
                if parties.get("respondent"):
                    lines.append(f"  Respondent       : {parties.get('respondent')}")
                if parties.get("insurance_company"):
                    lines.append(f"  Insurer Involved : {parties.get('insurance_company')}")

            # Legal Sections Cited
            sections = r.get("legal_sections", {})
            if sections:
                sec_list = []
                for act, s_list in sections.items():
                    sec_list.append(f"{act} ({', '.join(s_list[:4])})")
                lines.append(f"  Statutes / Sec.  : {'; '.join(sec_list)}")

            # Historical Decision & Evidence
            lines.append(f"  Historical Outcome: {r.get('decision')}")
            evidence = r.get("decision_evidence")
            if evidence:
                # Truncate evidence cleanly
                clean_ev = evidence.replace("\n", " ").strip()
                if len(clean_ev) > 220:
                    clean_ev = clean_ev[:217] + "..."
                lines.append(f"  Operative Evidence: \"{clean_ev}\"")

            lines.append(f"  Quality Score    : {r.get('data_quality_score', 0.0):.2f}/1.00")
            lines.append(f"  Source URL       : {r.get('source_url') or 'N/A'}\n")

        lines.append("=" * 70)
        lines.append("[NOTE] These retrieved historical precedents serve as the factual and")
        lines.append("       legal evidence for downstream multi-case outcome reasoning.")
        lines.append("=" * 70 + "\n")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve Semantically Similar Validated Legal Precedents using FAISS"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Description of the new legal case, facts, accident details, or legal issues"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest historical cases to retrieve (default: 3)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON array instead of formatted text cards"
    )

    args = parser.parse_args()

    try:
        retriever = LegalCaseRetriever()
    except Exception as e:
        print(f"[ERROR] Failed to initialize retriever: {e}")
        sys.exit(1)

    query_text = args.query
    if not query_text:
        print("\n" + "=" * 65)
        print("LEGAL PRECEDENT SEMANTIC SEARCH (Motor Accident Cases)")
        print("=" * 65)
        query_text = input("\nEnter facts or legal issues of the new case:\n").strip()

    if not query_text:
        print("[WARN] No query entered. Exiting.")
        return

    results = retriever.search(query_text=query_text, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(retriever.format_results_as_text(results, query_text))


if __name__ == "__main__":
    main()
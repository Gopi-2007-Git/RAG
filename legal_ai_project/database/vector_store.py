"""
Vector Database Builder using FAISS IndexFlatL2 for Legal Precedent Retrieval.
Builds an exact Euclidean distance index exclusively from the validated Prediction Dataset,
maintaining a strict 1-to-1 mapping between vector positions, metadata, and historical outcomes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np
import pandas as pd

from pipeline.config import (
    DATABASE_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    PREDICTION_DATASET_CSV,
    PIPELINE_LOG_FILE
)
from pipeline.embedding import LegalEmbedder

logger = logging.getLogger(__name__)

INDEX_FILE = DATABASE_DIR / "legal_cases.index"
CASE_NAMES_FILE = DATABASE_DIR / "case_names.txt"
CASE_MAPPING_FILE = DATABASE_DIR / "case_mapping.json"


class LegalVectorKnowledgeBase:
    """Manages creation, indexing, and mapping for the validated legal knowledge base."""

    def __init__(
        self,
        index_file: Path = INDEX_FILE,
        case_names_file: Path = CASE_NAMES_FILE,
        case_mapping_file: Path = CASE_MAPPING_FILE
    ):
        self.index_file = Path(index_file)
        self.case_names_file = Path(case_names_file)
        self.case_mapping_file = Path(case_mapping_file)
        self.embedder = LegalEmbedder()

    def get_validated_case_ids(self) -> List[str]:
        """
        Load only cases that passed quality validation and are in the prediction dataset.
        Excludes all interim orders, procedural orders, and unvalidated documents.
        """
        if not PREDICTION_DATASET_CSV.exists():
            logger.error("Prediction dataset CSV not found at %s", PREDICTION_DATASET_CSV)
            return []

        df = pd.read_csv(PREDICTION_DATASET_CSV)
        if df.empty or "case_id" not in df.columns:
            logger.warning("Prediction dataset is empty or malformed.")
            return []

        # Filter strictly for usable cases
        usable_cases = df[df["processing_status"] == "usable"]["case_id"].dropna().tolist()
        logger.info("Loaded %d validated usable case(s) from prediction dataset.", len(usable_cases))
        return sorted(usable_cases)

    def load_case_metadata(self, case_id: str) -> Dict[str, Any]:
        """Load rich structured metadata for a specific case ID."""
        meta_file = METADATA_DIR / f"{case_id}.json"
        if not meta_file.exists():
            return {"case_id": case_id}

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error reading metadata for %s: %s", case_id, e)
            return {"case_id": case_id}

    def build_or_rebuild(self, force_reembed: bool = False) -> Tuple[int, int]:
        """
        Build or rebuild the FAISS IndexFlatL2 vector database and case mapping registry.
        Returns: (total_indexed, vector_dimension)
        """
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        validated_case_ids = self.get_validated_case_ids()

        if not validated_case_ids:
            print("[ERROR] No validated prediction cases found to index.")
            return 0, 0

        embeddings_list = []
        indexed_case_names = []
        mapping_records = []

        print("\n" + "=" * 65)
        print("BUILDING VALIDATED LEGAL KNOWLEDGE BASE (FAISS IndexFlatL2)")
        print("=" * 65)
        print(f"Target validated cases : {len(validated_case_ids)}")
        print(f"Embedding model        : {self.embedder.model_name} (384 dimensions)")
        print(f"Distance metric        : Euclidean L2 (IndexFlatL2)")
        print("=" * 65 + "\n")

        for idx, case_id in enumerate(validated_case_ids):
            embedding_file = PROCESSED_DIR / f"{case_id}_embedding.npy"

            # Generate embedding if missing or if forced
            if not embedding_file.exists() or force_reembed:
                print(f"[{idx + 1}/{len(validated_case_ids)}] Generating embedding for {case_id}...")
                res = self.embedder.embed_case(case_id)
                if not res:
                    logger.warning("Skipping %s due to embedding failure.", case_id)
                    continue

            try:
                vec = np.load(embedding_file).astype("float32")
                if vec.ndim == 1:
                    vec = np.expand_dims(vec, axis=0)

                embeddings_list.append(vec[0])
                indexed_case_names.append(case_id)

                # Fetch full metadata to build mapping record
                meta = self.load_case_metadata(case_id)
                mapping_record = {
                    "index_position": len(indexed_case_names) - 1,
                    "case_id": case_id,
                    "case_title": meta.get("case_title", "Unknown Title"),
                    "court": meta.get("court", "Unknown Court"),
                    "judgment_date": meta.get("judgment_date", "Unknown Date"),
                    "case_number": meta.get("case_number", "N/A"),
                    "parties": meta.get("parties", {}),
                    "bench": meta.get("bench", []),
                    "legal_sections": meta.get("legal_sections", {}),
                    "citations": meta.get("citations", []),
                    "decision": meta.get("decision", "not_identified"),
                    "decision_evidence": meta.get("decision_evidence") or "",
                    "source_url": meta.get("source_url", ""),
                    "data_quality_score": meta.get("data_quality_score", 0.0),
                    "raw_text_length": meta.get("raw_text_length", 0),
                    "cleaned_text_length": meta.get("cleaned_text_length", 0)
                }
                mapping_records.append(mapping_record)
                print(f"[{idx + 1}/{len(validated_case_ids)}] [OK] Indexed {case_id} -> Outcome: {mapping_record['decision']}")

            except Exception as e:
                logger.error("Error reading embedding for %s: %s", case_id, e)

        if not embeddings_list:
            print("[ERROR] Failed to load any embeddings for indexing.")
            return 0, 0

        # Create FAISS IndexFlatL2
        embeddings_matrix = np.array(embeddings_list).astype("float32")
        dimension = embeddings_matrix.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_matrix)

        # Persist FAISS index file
        faiss.write_index(index, str(self.index_file))

        # Persist case names text file
        with open(self.case_names_file, "w", encoding="utf-8") as f:
            for name in indexed_case_names:
                f.write(name + "\n")

        # Persist full case mapping registry JSON
        with open(self.case_mapping_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.embedder.model_name,
                "dimension": dimension,
                "distance_metric": "IndexFlatL2",
                "total_cases": len(mapping_records),
                "cases": mapping_records
            }, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 65)
        print("[OK] KNOWLEDGE BASE & FAISS INDEX CREATED SUCCESSFULLY")
        print("=" * 65)
        print(f"Total Validated Cases Indexed : {index.ntotal}")
        print(f"Vector Embedding Dimension    : {dimension}")
        print(f"FAISS Index File              : {self.index_file}")
        print(f"Case Names Registry           : {self.case_names_file}")
        print(f"Case Mapping Registry JSON    : {self.case_mapping_file}")
        print("=" * 65 + "\n")

        return index.ntotal, dimension


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build FAISS Vector Database for Validated Legal Cases")
    parser.add_argument("--rebuild", action="store_true", help="Force recalculating embeddings and index")
    args = parser.parse_args()

    kb = LegalVectorKnowledgeBase()
    kb.build_or_rebuild(force_reembed=args.rebuild)


if __name__ == "__main__":
    main()
    
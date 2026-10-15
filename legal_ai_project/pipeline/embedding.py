"""
Embedding Generator Agent for Prediction Dataset Judgments.
Generates dense vector embeddings using Sentence Transformers with text chunking
for long legal texts, enabling high-accuracy semantic search and precedent retrieval.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    PROCESSED_DIR,
    DATASET_DIR,
    PREDICTION_DATASET_CSV,
    PIPELINE_LOG_FILE
)

logger = logging.getLogger(__name__)


class LegalEmbedder:
    """Generates dense vector embeddings for legal judgments."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_chunk_tokens: int = 250):
        self.model_name = model_name
        self.max_chunk_tokens = max_chunk_tokens
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split long legal judgment text into overlapping word chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += (chunk_size - overlap)

        return chunks

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed full judgment text. If text is long, averages chunk embeddings.
        """
        chunks = self.chunk_text(text)
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        # Average pool chunk embeddings for a representative document vector
        doc_embedding = np.mean(chunk_embeddings, axis=0)
        # Re-normalize vector
        norm = np.linalg.norm(doc_embedding)
        if norm > 0:
            doc_embedding = doc_embedding / norm
        return doc_embedding.astype("float32")

    def embed_case(self, case_id: str) -> Optional[Path]:
        """Generate and save embedding for a single processed case."""
        proc_file = PROCESSED_DIR / f"{case_id}.txt"
        if not proc_file.exists():
            clean_name = case_id.replace("case_", "")
            alt = PROCESSED_DIR / f"{clean_name}.txt"
            if alt.exists():
                proc_file = alt
            else:
                logger.error("Processed file not found: %s", proc_file)
                return None

        text = proc_file.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            logger.warning("Empty processed text for %s", case_id)
            return None

        embedding = self.embed_text(text)
        out_file = PROCESSED_DIR / f"{case_id}_embedding.npy"
        np.save(out_file, embedding)
        logger.info("Generated embedding for %s (shape: %s) -> %s", case_id, embedding.shape, out_file.name)
        return out_file

    def embed_prediction_dataset(self) -> int:
        """Embed all usable cases in the prediction dataset."""
        import pandas as pd
        if not PREDICTION_DATASET_CSV.exists():
            logger.warning("Prediction dataset CSV not found: %s", PREDICTION_DATASET_CSV)
            return 0

        df = pd.read_csv(PREDICTION_DATASET_CSV)
        if df.empty:
            logger.warning("Prediction dataset is empty.")
            return 0

        success_count = 0
        for _, row in df.iterrows():
            case_id = row["case_id"]
            res = self.embed_case(case_id)
            if res:
                success_count += 1

        logger.info("Embedded %d/%d prediction dataset cases.", success_count, len(df))
        return success_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Generate Sentence Transformer embeddings for legal cases")
    parser.add_argument("case_file", nargs="?", help="Specific case ID to embed (optional)")
    parser.add_argument("--batch-prediction", action="store_true", help="Embed all usable prediction cases")
    args = parser.parse_args()

    embedder = LegalEmbedder()

    if args.case_file:
        clean_id = args.case_file.replace(".txt", "").replace(".npy", "").replace("_embedding", "")
        if not clean_id.startswith("case_") and clean_id.isdigit():
            clean_id = f"case_{clean_id}"
        embedder.embed_case(clean_id)
    elif args.batch_prediction:
        embedder.embed_prediction_dataset()
    else:
        # Default: embed all processed files
        txt_files = list(PROCESSED_DIR.glob("*.txt"))
        for tf in txt_files:
            embedder.embed_case(tf.stem)
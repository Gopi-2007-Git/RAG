"""
Data Collection Agent for Indian Kanoon Legal Judgments.
Automates search query submission, document URL discovery, pagination,
rate-limited fetching, duplicate detection, and raw storage with source traceability.
"""

import argparse
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from bs4 import BeautifulSoup

from pipeline.config import (
    BASE_URL,
    SEARCH_ENDPOINT,
    DOC_ENDPOINT,
    DEFAULT_QUERY,
    DEFAULT_MAX_DOCS,
    DEFAULT_MAX_PAGES,
    REQUEST_TIMEOUT,
    REQUEST_DELAY,
    DELAY_JITTER_MIN,
    DELAY_JITTER_MAX,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    HTTP_HEADERS,
    MIN_TEXT_LENGTH,
    RAW_DIR,
    METADATA_DIR,
    PIPELINE_LOG_FILE,
    ERROR_LOG_FILE
)
from pipeline.duplicate_detector import DuplicateDetector

# Configure Module Logger
logger = logging.getLogger(__name__)


class IndianKanoonCollector:
    """Automated legal judgment collection pipeline for Indian Kanoon."""

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        max_docs: int = DEFAULT_MAX_DOCS,
        max_pages: int = DEFAULT_MAX_PAGES,
        delay: float = REQUEST_DELAY,
        duplicate_detector: Optional[DuplicateDetector] = None
    ):
        self.query = query
        self.max_docs = max_docs
        self.max_pages = max_pages
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HTTP_HEADERS)
        self.detector = duplicate_detector or DuplicateDetector()
        self.stats = {
            "query": self.query,
            "discovered_urls": 0,
            "collected": 0,
            "duplicates_skipped": 0,
            "insufficient_text_skipped": 0,
            "http_errors": 0,
            "failed_urls": []
        }

    def _sleep_with_jitter(self) -> None:
        """Rate limiting sleep with random jitter to avoid aggressive scraping."""
        jitter = random.uniform(DELAY_JITTER_MIN, DELAY_JITTER_MAX)
        total_sleep = max(0.5, self.delay + jitter)
        time.sleep(total_sleep)

    def _extract_doc_id(self, url_or_path: str) -> Optional[str]:
        """Extract canonical document ID from Indian Kanoon URL/path."""
        match = re.search(r"/doc(?:fragment)?/(\d+)", url_or_path)
        if match:
            return match.group(1)
        return None

    def search_page(self, page_num: int = 0) -> List[Dict[str, str]]:
        """
        Execute search on Indian Kanoon and discover document links.
        Returns a list of dicts with doc_id, url, and preliminary title.
        """
        encoded_query = quote(self.query)
        if page_num == 0:
            url = f"{SEARCH_ENDPOINT}?formInput={encoded_query}"
        else:
            url = f"{SEARCH_ENDPOINT}?formInput={encoded_query}&pagenum={page_num}"

        logger.info("Searching Indian Kanoon [Page %d]: %s", page_num + 1, url)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    break
                elif response.status_code in (429, 503):
                    wait = BACKOFF_FACTOR ** attempt + random.uniform(1.0, 2.0)
                    logger.warning("HTTP %d on search page %d. Backing off for %.1fs (attempt %d/%d)",
                                   response.status_code, page_num + 1, wait, attempt, MAX_RETRIES)
                    time.sleep(wait)
                else:
                    logger.error("HTTP %d while searching Indian Kanoon: %s", response.status_code, url)
                    self.stats["http_errors"] += 1
                    return []
            except requests.RequestException as e:
                logger.warning("Connection error searching page %d (attempt %d/%d): %s",
                               page_num + 1, attempt, MAX_RETRIES, e)
                if attempt == MAX_RETRIES:
                    self.stats["http_errors"] += 1
                    return []
                time.sleep(BACKOFF_FACTOR ** attempt)

        soup = BeautifulSoup(response.text, "html.parser")
        discovered = []

        # Find judgment links
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            doc_id = self._extract_doc_id(href)
            if doc_id:
                canonical_url = f"{DOC_ENDPOINT}{doc_id}/"
                title_text = a_tag.get_text(strip=True) or f"Document {doc_id}"
                if not any(d["doc_id"] == doc_id for d in discovered):
                    discovered.append({
                        "doc_id": doc_id,
                        "case_id": f"case_{doc_id}",
                        "url": canonical_url,
                        "search_title": title_text
                    })

        logger.info("Found %d judgment links on page %d", len(discovered), page_num + 1)
        return discovered

    def fetch_document(self, doc_url: str) -> Optional[Tuple[str, str]]:
        """
        Fetch full judgment page from Indian Kanoon with retry and rate-limiting.
        Returns: (raw_text, raw_html) or None if failed.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._sleep_with_jitter()
                response = self.session.get(doc_url, timeout=REQUEST_TIMEOUT)

                if response.status_code == 200:
                    raw_html = response.text
                    soup = BeautifulSoup(raw_html, "html.parser")
                    raw_text = soup.get_text("\n", strip=True)
                    return raw_text, raw_html
                elif response.status_code in (429, 503):
                    wait = BACKOFF_FACTOR ** attempt + random.uniform(2.0, 4.0)
                    logger.warning("Rate limit / server busy (HTTP %d). Waiting %.1fs...",
                                   response.status_code, wait)
                    time.sleep(wait)
                else:
                    logger.warning("HTTP %d for document: %s", response.status_code, doc_url)
                    self.stats["http_errors"] += 1
                    return None
            except requests.RequestException as e:
                logger.warning("Request failed for %s (attempt %d/%d): %s",
                               doc_url, attempt, MAX_RETRIES, e)
                if attempt == MAX_RETRIES:
                    self.stats["http_errors"] += 1
                    self.stats["failed_urls"].append(doc_url)
                    return None
                time.sleep(BACKOFF_FACTOR ** attempt)

        return None

    def save_raw_document(
        self,
        case_id: str,
        doc_id: str,
        raw_text: str,
        source_url: str,
        search_title: str
    ) -> Path:
        """Save raw document text and create base metadata file."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_file = RAW_DIR / f"{case_id}.txt"

        # Save raw text
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Register with duplicate detector
        self.detector.register(case_id, raw_text)

        logger.info("Saved raw judgment: %s (%d chars)", raw_file.name, len(raw_text))
        return raw_file

    def collect(self) -> Dict:
        """
        Execute discovery and collection workflow.
        Returns a dictionary of execution statistics.
        """
        logger.info("=" * 60)
        logger.info("STARTING AUTOMATIC LEGAL JUDGMENT COLLECTION")
        logger.info("Domain / Query : %s", self.query)
        logger.info("Target Cases   : %d", self.max_docs)
        logger.info("Max Pages      : %d", self.max_pages)
        logger.info("=" * 60)

        all_discovered = []
        page = 0

        # Step 1: Multi-page search discovery
        while len(all_discovered) < self.max_docs and page < self.max_pages:
            page_results = self.search_page(page)
            if not page_results:
                logger.info("No more search results found on page %d.", page + 1)
                break

            for doc_info in page_results:
                if not any(d["doc_id"] == doc_info["doc_id"] for d in all_discovered):
                    all_discovered.append(doc_info)
                    if len(all_discovered) >= self.max_docs:
                        break

            page += 1
            self._sleep_with_jitter()

        self.stats["discovered_urls"] = len(all_discovered)
        logger.info("Discovered %d candidate judgment URLs across %d page(s)",
                    len(all_discovered), page)

        # Step 2: Fetch and Store Each Case
        for idx, doc_info in enumerate(all_discovered, start=1):
            if self.stats["collected"] >= self.max_docs:
                break

            case_id = doc_info["case_id"]
            doc_id = doc_info["doc_id"]
            url = doc_info["url"]

            logger.info("\n[%d/%d] Processing: %s (Doc ID: %s)",
                        idx, len(all_discovered), url, doc_id)

            # Pre-fetch Duplicate Check by ID
            if self.detector.is_duplicate_id(case_id):
                logger.info("-> Skipped: Case ID %s already collected locally.", case_id)
                self.stats["duplicates_skipped"] += 1
                continue

            # Fetch Document
            fetch_result = self.fetch_document(url)
            if not fetch_result:
                logger.warning("-> Failed to fetch document: %s", url)
                continue

            raw_text, _ = fetch_result

            # Check Minimum Length Threshold
            if len(raw_text) < MIN_TEXT_LENGTH:
                logger.warning("-> Skipped: Insufficient text length (%d chars < %d required)",
                               len(raw_text), MIN_TEXT_LENGTH)
                self.stats["insufficient_text_skipped"] += 1
                continue

            # Post-fetch Duplicate Content Check by Hash
            is_dup_content, existing_match = self.detector.is_duplicate_content(raw_text)
            if is_dup_content:
                logger.info("-> Skipped: Duplicate content hash matching %s", existing_match)
                self.stats["duplicates_skipped"] += 1
                continue

            # Save Document
            self.save_raw_document(
                case_id=case_id,
                doc_id=doc_id,
                raw_text=raw_text,
                source_url=url,
                search_title=doc_info["search_title"]
            )
            self.stats["collected"] += 1

        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION SUMMARY")
        logger.info("Judgment links discovered : %d", self.stats["discovered_urls"])
        logger.info("New documents collected    : %d", self.stats["collected"])
        logger.info("Duplicates skipped         : %d", self.stats["duplicates_skipped"])
        logger.info("Insufficient text skipped  : %d", self.stats["insufficient_text_skipped"])
        logger.info("HTTP / Fetch errors        : %d", self.stats["http_errors"])
        logger.info("=" * 60)

        return self.stats


def setup_logging():
    """Configure console and file logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
        ]
    )


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Indian Kanoon Legal Judgment Collector")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Search query or domain")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Max cases to collect")
    parser.add_argument("--pages", type=int, default=DEFAULT_MAX_PAGES, help="Max search pages to check")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between requests in seconds")

    args = parser.parse_args()

    collector = IndianKanoonCollector(
        query=args.query,
        max_docs=args.max_docs,
        max_pages=args.pages,
        delay=args.delay
    )
    collector.collect()
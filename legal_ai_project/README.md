# AI-Powered Indian Legal Case Prediction System
## Automated Legal Judgment Dataset Collection & Processing Pipeline

A modular, automated, fault-tolerant dataset collection and curation pipeline designed to discover, fetch, clean, validate, and organize historical Indian court judgments from **Indian Kanoon**. 

The initial pipeline specializes in the **Motor Accident Domain** (Motor Vehicles Act 1988, MACT compensation, insurer liability, composite/contributory negligence, and quantum enhancement/reduction), with modular support to adapt to any Indian legal domain without rewriting code.

---

## 🏛️ System Architecture

```
                    Indian Kanoon Search Engine
                                 │
                 ┌───────────────┴───────────────┐
                 │  1. Automatic Discovery       │
                 │  (Multi-page pagination, rate │
                 │   limits, jitter & backoff)   │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  2. Deduplication Layer       │
                 │  (Kanoon Doc ID + SHA-256)    │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  3. Raw Storage               │
                 │  (data/raw/case_<id>.txt)     │
                 └───────────────┬───────────────┘
                                 ▼
        ┌────────────────────────┼────────────────────────┐
        ▼                                                 ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│ 4. Metadata Extraction Agent  │         │ 5. Document Cleaning Agent    │
│ Title, Court, Date, Case No., │         │ Strips Prism AI, navigation,  │
│ Bench, Parties, MVA Sections, │         │ ads; preserves legal facts,   │
│ Citations & Source URLs       │         │ reasoning & operative orders  │
└───────────────┬───────────────┘         └───────────────┬───────────────┘
                │                                         │
                └────────────────┬────────────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │ 6. Decision & Doc Classifier  │
                 │ Classifies Final vs Interim   │
                 │ Normalizes multi-expression   │
                 │ outcomes & extracts evidence  │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │ 7. Quality Validation Agent   │
                 │ Length, metadata validity,    │
                 │ outcome certainty & score     │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     [Candidate Dataset]               [Prediction Dataset]
     (All collected cases)             (High-quality usable cases)
                 │                               │
                 ▼                               ▼
        data/dataset/manifest.csv       data/dataset/prediction_dataset.csv
                                                 │
                                                 ▼
                                        Sentence Transformers
                                                 ▼
                                        FAISS Vector Database
                                                 ▼
                                        Precedent Retrieval &
                                        Outcome Prediction
```

---

## 📂 Project Directory Structure

```
legal_ai_project/
├── data/
│   ├── raw/                  # Traceable raw judgment texts (case_<id>.txt)
│   ├── metadata/             # Structured JSON metadata (case_<id>.json)
│   ├── processed/            # Cleaned judgments preserving legal text (case_<id>.txt)
│   ├── excluded/             # Excluded records with diagnostic reason JSON
│   └── dataset/              # Manifests and curated prediction datasets
│       ├── manifest.csv               # Unified dataset manifest
│       ├── manifest.jsonl             # JSON Lines format manifest
│       ├── prediction_dataset.csv     # Filtered usable cases for AI prediction
│       ├── prediction_dataset.jsonl   # Filtered usable cases (JSONL)
│       ├── candidate_dataset.csv      # All collected cases
│       ├── hash_registry.json         # SHA-256 deduplication registry
│       └── summary_report.json        # Dataset statistics & metrics
├── database/                 # FAISS vector store & case index
│   ├── legal_cases.index
│   ├── case_names.txt
│   └── vector_store.py
├── logs/                     # Execution and error logs
│   ├── pipeline.log
│   └── pipeline_errors.log
├── pipeline/                 # Core modular pipeline components
│   ├── __init__.py
│   ├── config.py             # Centralized settings, timeouts, thresholds
│   ├── collector.py          # Discovery & polite crawling agent
│   ├── duplicate_detector.py # Kanoon ID & SHA-256 content hashing
│   ├── metadata_extractor.py # Court, title, date, citations & sections
│   ├── cleaner.py            # Boilerplate, promo & UI noise remover
│   ├── decision_extractor.py # Document type & operative decision parser
│   ├── validator.py          # Quality scoring (0-1.0) & dataset routing
│   ├── dataset_builder.py    # Manifest generator & summary statistics
│   ├── process_all.py        # Master pipeline orchestrator
│   ├── embedding.py          # Sentence Transformer dense embeddings
│   └── search_cases.py       # Semantic precedent retrieval agent
├── tests/                    # Comprehensive unit test suite
│   ├── test_duplicate_detector.py
│   ├── test_cleaner.py
│   ├── test_metadata_extractor.py
│   ├── test_decision_extractor.py
│   ├── test_validator.py
│   └── test_dataset_builder.py
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## ⚙️ Key Features

### 1. Automated Discovery & Polite Collection (`collector.py`)
- Searches Indian Kanoon across multiple pages (`pagenum=0, 1, 2...`).
- Implements rate limiting (2.0s base + random jitter 0.5s–1.5s).
- Implements exponential backoff retries on HTTP 429/503.
- Prevents re-downloading existing cases using pre-fetch ID checks.

### 2. Multi-level Deduplication (`duplicate_detector.py`)
- **ID Matching**: Detects canonical Indian Kanoon document IDs.
- **Content Hashing**: SHA-256 hash of normalized judgment text to catch duplicate uploads under different URLs.
- Persistent index at `data/dataset/hash_registry.json`.

### 3. Structured Metadata Extraction (`metadata_extractor.py`)
- **Courts**: Supreme Court of India, High Courts (Karnataka, Delhi, Madras, Bombay, etc.), MACT Tribunals.
- **Titles**: Extracts `"Party A vs Party B"` from case headers.
- **Judgment Dates**: Parses diverse formats into ISO standard `YYYY-MM-DD`.
- **Case Numbers**: Extracts `MACA`, `MFA`, `MVC`, `Civil Appeal`, `SLP`, `Writ Petition`.
- **Bench / Judges**: Extracts sitting judges and author names.
- **Statutes & Sections**: Focuses on Motor Vehicles Act 1988 (Sections 166, 163A, 173, 140, 149), IPC (279, 304A), and CPC provisions.
- **Equivalent Citations**: Captures citations (e.g. `2000 ACJ 1463`, `AIR 2001 SC 100`).

### 4. Document Cleaning & Legal Preservation (`cleaner.py`)
- Strips navigation menus, "Prism AI" banners, "Doc Gen Hub", "Case Predict AI", premium notices, PDF/Print buttons, and bottom AI tags.
- Normalizes unicode artifacts (smart quotes, em-dashes, non-breaking spaces).
- **Strictly preserves**: Facts, accident descriptions, FIR details, witness statements, disability percentages, salary calculations, multiplier application, insurer contentions, legal analysis, and operative final awards.

### 5. Document Classification & Decision Extraction (`decision_extractor.py`)
- Distinguishes **Final Judgments** from **Interim Orders**, **Procedural Orders**, and **Further Hearing Adjournments**.
- Extracts multi-expression domain outcomes:
  - `compensation_enhanced`
  - `compensation_reduced`
  - `appeal_allowed`, `appeal_dismissed`, `appeal_partly_allowed`
  - `petition_allowed`, `petition_dismissed`, `petition_partly_allowed`
  - `claim_allowed`, `claim_dismissed`, `claim_partly_allowed`
  - `order_set_aside`, `order_modified`, `order_confirmed`
  - `remanded`, `application_allowed`, `application_dismissed`
- Captures exact verbatim sentence as `decision_evidence`.
- Labels uncertain cases as `not_identified` to avoid false assumptions.

### 6. Quality Validation & Dataset Segregation (`validator.py`)
- Validates text length (>= 1000 chars, >= 150 words), valid URL, complete metadata, and outcome reliability.
- Computes `data_quality_score` (0.00 to 1.00).
- Splits records into:
  - **Prediction Dataset**: High-quality final judgments ready for AI outcome prediction.
  - **Candidate Dataset**: All collected records.
  - **Excluded Records**: Stored with diagnostic reasons in `data/excluded/{case_id}.json`.

### 5. Multi-Case Legal Outcome Analysis & AI Prediction (`predictor.py`)
- **Beyond Majority Voting**: Evaluates semantic proximity ($L_2$ distance), judicial authority (Supreme Court vs. High Court), statutory overlap (e.g. MVA 1988 Sections 166, 173), and factual congruence (disability, death, liability) rather than naive headcounts.
- **Evidence-Based Reasoning**: Traces findings back to exact historical case precedents with operative decision excerpts.
- **Consensus & Conflict Detection**: Identifies whether retrieved precedents align or conflict, evaluating evidence strength (`High`, `Moderate`, `Low`, `Insufficient`).
- **Explainability Report**: Renders predicted outcome, precedent cards, outcome distribution, legal similarities, factual differences, reasoning trace, limitations, and mandatory legal disclaimer.

---

## 🚀 Quickstart Guide

### 1. Installation
Install dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Run Unit Tests
Verify all components across all 3 phases (36 tests):
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Run the Automated Pipeline
Run end-to-end collection, cleaning, metadata extraction, decision parsing, and dataset generation:
```powershell
# Discover and collect motor accident cases from Indian Kanoon
python pipeline/process_all.py --query "motor accident doctypes:judgments" --max-docs 20 --pages 4
```

To re-process existing raw files locally without downloading:
```powershell
python pipeline/process_all.py --skip-collector
```

### 4. Build Embeddings & Semantic Search
Generate dense vectors and query similar cases:
```powershell
# Build FAISS vector database (automatically indexes all validated prediction cases)
python database/vector_store.py

# Search similar precedent cases
python pipeline/search_cases.py --query "claimant injured in motor accident seeking enhancement of compensation under section 173 MVA" --top-k 3
```

### 5. AI-Assisted Case Outcome Prediction
Predict the legal outcome for a new case using multi-precedent synthesis:
```powershell
# Interactive mode
python pipeline/predictor.py

# Direct CLI Query
python pipeline/predictor.py --query "The claimant suffered 60% permanent disability in a head-on collision between a bus and lorry. The tribunal awarded Rs. 1,50,000 which is inadequate. The claimant appeals under Section 173 of the Motor Vehicles Act seeking enhancement of compensation." --top-k 3

# JSON output mode for web application integration
python pipeline/predictor.py --query "Fatal road accident compensation dispute under Section 166 MVA" --top-k 3 --json
```

---

## 📋 Manifest Schema

The generated dataset manifests (`data/dataset/manifest.csv` and `data/dataset/prediction_dataset.csv`) contain the following fields:

| Column | Description | Example |
| :--- | :--- | :--- |
| `case_id` | Unique traceable identifier | `case_1501320` |
| `case_title` | Appellant vs Respondent | `KSRTC vs Reny Mammen And Others` |
| `court` | Adjudicating court | `Karnataka High Court` |
| `judgment_date` | Date of judgment (ISO) | `1990-08-31` |
| `case_number` | Formal appeal/claim number | `MVC Nos. 20 of 1985` |
| `source_url` | Traceable Indian Kanoon URL | `https://indiankanoon.org/doc/1501320/` |
| `document_type` | Document classification | `final_judgment` |
| `decision` | Normalized legal outcome | `appeal_partly_allowed` |
| `decision_evidence` | Verbatim operative snippet | `"The appeals are partly allowed..."` |
| `text_length` | Character count of judgment | `92759` |
| `collection_date` | Timestamp of collection | `2026-08-24T21:18:02.123456+00:00` |
| `processing_status`| Status (`usable` / `excluded`) | `usable` |
| `exclusion_reason` | Diagnostic reason if excluded | `None` / `interim_order` |
| `data_quality_score`| Composite quality metric | `0.90` |

---

## 🔄 Adapting to Other Legal Domains

To collect and train models on another legal domain (e.g., Land Acquisition, Dishonour of Cheques under NI Act, Consumer Disputes, Criminal Appeals), simply pass the new search query to `process_all.py`:

```powershell
# Dishonour of Cheque / Section 138 NI Act
python pipeline/process_all.py --query "section 138 negotiable instruments act doctypes:judgments" --max-docs 50

# Consumer Protection Act Cases
python pipeline/process_all.py --query "consumer protection medical negligence doctypes:judgments" --max-docs 50
```

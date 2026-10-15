"""
Multi-Case Legal Outcome Analysis and AI-Assisted Judgment Prediction Engine.
Synthesizes retrieved historical precedents, judicial hierarchy, statutory alignment,
and factual congruence to produce explainable, evidence-based legal outcome predictions.
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    USABLE_DECISION_OUTCOMES,
    PIPELINE_LOG_FILE
)
from pipeline.search_cases import LegalCaseRetriever

logger = logging.getLogger(__name__)

# Court Hierarchy Authority Weights
COURT_AUTHORITY_WEIGHTS = {
    "Supreme Court of India": 1.30,
    "High Court": 1.00,
    "Tribunal": 0.85,
    "Unknown Court": 0.80
}

# Factual Feature Extraction Keywords for Motor Accident Domain
FACT_KEYWORDS = {
    "fatal_or_death": [r"\bdeath\b", r"\bfatal\b", r"\bkilled\b", r"\bdeceased\b", r"\blegal\s+heirs\b", r"\bloss\s+of\s+dependency\b"],
    "permanent_disability": [r"\bdisability\b", r"\bpermanent\s+disability\b", r"\bdisabled\b", r"\bphysical\s+impairment\b", r"\bloss\s+of\s+earning\s+capacity\b"],
    "injury_claim": [r"\binjur(?:y|ies|ed)\b", r"\bgrievous\b", r"\bfracture\b", r"\bhospitaliz(?:ation|ed)\b", r"\bmedical\s+expenses\b"],
    "negligence_dispute": [r"\bnegligen(?:ce|t)\b", r"\brash\b", r"\bcomposite\s+negligence\b", r"\bcontributory\s+negligence\b", r"\bcollision\b", r"\bhead[\-\s]on\b"],
    "insurance_liability": [r"\binsur(?:ance|er|ed|ing)?\b", r"\bpolicy\b", r"\bbreach\s+of\s+policy\b", r"\bdriving\s+licen(?:ce|se)\b", r"\bpermit\b", r"\bgratuitous\s+passenger\b", r"\bthird\s+party\b"],
    "compensation_quantum": [r"\bcompensation\b", r"\benhance(?:ment|d)?\b", r"\breduc(?:tion|ed)?\b", r"\bquantum\b", r"\binadequate\b", r"\bmultiplier\b", r"\bfuture\s+prospects\b", r"\binterest\b"]
}

# Statutory Feature Extraction Rules
STATUTE_REGEXES = {
    "MVA 1988 - Sec 166 (Compensation Claim)": r"(?:section|sec\.?|u/s)\s*166\b|claim\s+petition\s+under\s+section\s+166",
    "MVA 1988 - Sec 163A (Structured Formula)": r"(?:section|sec\.?|u/s)\s*163-?A\b",
    "MVA 1988 - Sec 173 (Appeals)": r"(?:section|sec\.?|u/s)\s*173\b|appeal\s+under\s+section\s+173",
    "MVA 1988 - Sec 140 (No Fault Liability)": r"(?:section|sec\.?|u/s)\s*140\b|no[\-\s]fault\s+liability",
    "MVA 1988 - Sec 149 / 147 (Insurer Liability)": r"(?:section|sec\.?|u/s)\s*(?:149|147)\b",
    "MVA 1988 - Sec 168 (Tribunal Award)": r"(?:section|sec\.?|u/s)\s*168\b",
    "IPC - Sec 279 / 304A (Rash Driving / Death)": r"(?:section|sec\.?|u/s)\s*(?:279|304-?A)\b",
    "CPC - Civil Procedure Provisions": r"\b(?:cpc|code\s+of\s+civil\s+procedure|order\s+(?:41|xli))\b"
}

# Normalized Outcome Display Labels
OUTCOME_DISPLAY_NAMES = {
    "appeal_allowed": "Appeal Allowed",
    "appeal_dismissed": "Appeal Dismissed",
    "appeal_partly_allowed": "Appeal Partly Allowed",
    "petition_allowed": "Petition Allowed",
    "petition_dismissed": "Petition Dismissed",
    "petition_partly_allowed": "Petition Partly Allowed",
    "claim_allowed": "Claim Allowed",
    "claim_dismissed": "Claim Dismissed",
    "claim_partly_allowed": "Claim Partly Allowed",
    "compensation_enhanced": "Compensation Enhanced",
    "compensation_reduced": "Compensation Reduced",
    "order_set_aside": "Order Set Aside",
    "order_modified": "Order Modified",
    "order_confirmed": "Order Confirmed",
    "application_allowed": "Application Allowed",
    "application_dismissed": "Application Dismissed",
    "remanded": "Remanded to Tribunal",
    "unable_to_predict": "Unable to Predict (Insufficient / Conflicting Evidence)"
}


class LegalCasePredictor:
    """Multi-case legal outcome reasoning and prediction engine."""

    def __init__(self, retriever: Optional[LegalCaseRetriever] = None):
        self.retriever = retriever or LegalCaseRetriever()

    @staticmethod
    def get_court_authority(court_name: Optional[str]) -> float:
        """Assign judicial hierarchy weight based on court level."""
        if not court_name:
            return COURT_AUTHORITY_WEIGHTS["Unknown Court"]
        c_lower = court_name.lower()
        if "supreme court" in c_lower:
            return COURT_AUTHORITY_WEIGHTS["Supreme Court of India"]
        elif "high court" in c_lower:
            return COURT_AUTHORITY_WEIGHTS["High Court"]
        elif "tribunal" in c_lower or "mact" in c_lower:
            return COURT_AUTHORITY_WEIGHTS["Tribunal"]
        return COURT_AUTHORITY_WEIGHTS["Unknown Court"]

    def extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """Extract factual themes and cited legal provisions from case text."""
        text_lower = text.lower()

        # Extract Factual Themes
        facts_found = []
        for theme, patterns in FACT_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    facts_found.append(theme)
                    break

        # Extract Cited Statutes
        statutes_found = []
        for statute_label, regex_pat in STATUTE_REGEXES.items():
            if re.search(regex_pat, text_lower):
                statutes_found.append(statute_label)

        return {
            "facts": set(facts_found),
            "statutes": set(statutes_found),
            "char_count": len(text),
            "word_count": len(text.split())
        }

    def compare_precedent(
        self,
        new_features: Dict[str, Any],
        precedent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare new case features against a retrieved historical precedent.
        Identifies factual overlap, statutory matches, and important distinctions.
        """
        hist_title = precedent.get("case_title", "")
        hist_evidence = precedent.get("decision_evidence", "")
        hist_sections = precedent.get("legal_sections", {})
        court_name = precedent.get("court", "Unknown Court")

        # Compile historical text representation for feature matching
        hist_text = f"{hist_title} {hist_evidence} "
        for act, sec_list in hist_sections.items():
            hist_text += f"{act} {' '.join(sec_list)} "

        hist_features = self.extract_features_from_text(hist_text)

        # Factual similarities & differences
        shared_facts = new_features["facts"].intersection(hist_features["facts"])
        new_unique_facts = new_features["facts"] - hist_features["facts"]
        hist_unique_facts = hist_features["facts"] - new_features["facts"]

        # Statute matches
        shared_statutes = new_features["statutes"].intersection(hist_features["statutes"])

        # Check sections extracted directly in metadata
        direct_section_matches = []
        for act, sec_list in hist_sections.items():
            for sec in sec_list:
                for new_statute in new_features["statutes"]:
                    if any(num in new_statute for num in re.findall(r"\d+", sec)):
                        direct_section_matches.append(f"{act} ({sec})")

        # Compute overlap scores
        fact_match_score = len(shared_facts) / max(1, len(new_features["facts"]))
        statute_match_score = (len(shared_statutes) + len(direct_section_matches)) > 0

        # Human-readable comparison items
        similarities = []
        if "fatal_or_death" in shared_facts:
            similarities.append("Involves fatal road accident and dependency compensation claim")
        if "permanent_disability" in shared_facts:
            similarities.append("Involves permanent disability and loss of earning capacity")
        if "injury_claim" in shared_facts:
            similarities.append("Involves bodily injury and medical expense adjudication")
        if "negligence_dispute" in shared_facts:
            similarities.append("Involves driver negligence and collision liability dispute")
        if "insurance_liability" in shared_facts:
            similarities.append("Involves insurance policy coverage and indemnification liability")
        if "compensation_quantum" in shared_facts:
            similarities.append("Involves quantum assessment, multiplier application, or enhancement")
        if shared_statutes or direct_section_matches:
            all_sec = list(shared_statutes) + direct_section_matches
            similarities.append(f"Shares applicable legal provisions: {', '.join(set(all_sec[:3]))}")

        # Default similarity if none matched specifically
        if not similarities:
            similarities.append("General motor accident claim and compensation dispute under Motor Vehicles Act")

        differences = []
        if "permanent_disability" in new_unique_facts:
            differences.append("New case involves permanent disability, which may differ from precedent's injury profile")
        if "fatal_or_death" in new_unique_facts:
            differences.append("New case involves fatal accident whereas precedent may focus primarily on non-fatal injury")
        if "insurance_liability" in hist_unique_facts:
            differences.append("Precedent extensively addresses insurer policy breaches which may not apply to new case")

        if not differences:
            differences.append("Differences in specific accident geometry, age/income of victim, and quantum awarded by Tribunal")

        return {
            "shared_facts": list(shared_facts),
            "shared_statutes": list(shared_statutes),
            "fact_match_score": fact_match_score,
            "statute_match_score": float(statute_match_score),
            "court_authority": self.get_court_authority(court_name),
            "similarities": similarities,
            "differences": differences
        }

    def synthesize_outcome(
        self,
        new_features: Dict[str, Any],
        precedents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize multi-precedent evidence to determine predicted outcome and evidence strength.
        Uses distance weighting, court hierarchy, and legal alignment.
        """
        if not precedents:
            return {
                "predicted_outcome": "unable_to_predict",
                "predicted_outcome_display": OUTCOME_DISPLAY_NAMES["unable_to_predict"],
                "evidence_strength": "Insufficient",
                "outcome_scores": {},
                "outcome_distribution": {},
                "synthesis_reasoning": "No relevant historical precedents were retrieved from the knowledge base."
            }

        # Check if the nearest precedent is too distant (L2 distance threshold)
        nearest_distance = precedents[0]["distance"]
        has_domain_features = bool(new_features.get("facts") or new_features.get("statutes"))

        # If zero domain facts/statutes and distance > 1.45, or any query with distance > 1.80
        if (not has_domain_features and nearest_distance > 1.45) or (nearest_distance > 1.80):
            return {
                "predicted_outcome": "unable_to_predict",
                "predicted_outcome_display": OUTCOME_DISPLAY_NAMES["unable_to_predict"],
                "evidence_strength": "Insufficient",
                "outcome_scores": {},
                "outcome_distribution": Counter(p["decision"] for p in precedents),
                "synthesis_reasoning": (
                    f"The query has insufficient domain relevance to the indexed motor accident precedents "
                    f"(Nearest L2 Distance: {nearest_distance:.4f}). "
                    "Semantic similarity and factual overlap are insufficient for a reliable legal outcome prediction."
                )
            }

        outcome_weights: Dict[str, float] = {}
        outcome_case_counts: Dict[str, int] = Counter()
        analyzed_precedents = []

        for p in precedents:
            decision = p.get("decision", "not_identified")
            dist = p.get("distance", 1.0)

            # Precedent comparison
            comp = self.compare_precedent(new_features, p)
            p_analyzed = dict(p)
            p_analyzed["comparison"] = comp
            analyzed_precedents.append(p_analyzed)

            if decision not in USABLE_DECISION_OUTCOMES:
                continue

            # Weight calculation:
            # w = (1 / (1 + d^2)) * CourtAuthority * (1 + 0.3*StatuteMatch + 0.2*FactMatch)
            base_dist_weight = 1.0 / (1.0 + (dist ** 2))
            authority_mult = comp["court_authority"]
            alignment_mult = 1.0 + (0.3 * comp["statute_match_score"]) + (0.2 * comp["fact_match_score"])

            case_weight = base_dist_weight * authority_mult * alignment_mult

            outcome_weights[decision] = outcome_weights.get(decision, 0.0) + case_weight
            outcome_case_counts[decision] += 1

        if not outcome_weights:
            return {
                "predicted_outcome": "unable_to_predict",
                "predicted_outcome_display": OUTCOME_DISPLAY_NAMES["unable_to_predict"],
                "evidence_strength": "Insufficient",
                "outcome_scores": {},
                "outcome_distribution": dict(outcome_case_counts),
                "analyzed_precedents": analyzed_precedents,
                "synthesis_reasoning": "Retrieved precedents do not contain verified definitive historical outcomes."
            }

        # Sort outcomes by weighted score
        sorted_outcomes = sorted(outcome_weights.items(), key=lambda x: x[1], reverse=True)
        top_outcome, top_score = sorted_outcomes[0]
        total_weight = sum(outcome_weights.values())
        top_support_ratio = top_score / total_weight if total_weight > 0 else 0.0

        # Assess Consensus vs Conflict
        has_strong_conflict = False
        if len(sorted_outcomes) > 1:
            runner_up_outcome, runner_up_score = sorted_outcomes[1]
            # If runner-up score is close (>80% of top score) and distance is very close
            if runner_up_score >= 0.82 * top_score and len(precedents) >= 2:
                has_strong_conflict = True

        # Determine Evidence Strength
        if has_strong_conflict:
            strength = "Low (Conflicting Precedents)"
        elif nearest_distance <= 0.85 and top_support_ratio >= 0.65:
            strength = "High"
        elif nearest_distance <= 1.15 and top_support_ratio >= 0.50:
            strength = "Moderate"
        else:
            strength = "Low"

        # If strong conflict with very small dataset, handle transparently
        predicted_outcome = top_outcome
        if has_strong_conflict and top_support_ratio < 0.55:
            predicted_outcome = top_outcome  # Retain top weighted, but flag in strength & reasoning

        # Formulate Detailed Legal Synthesis Reasoning
        top_cases = [p for p in analyzed_precedents if p.get("decision") == top_outcome]
        opposing_cases = [p for p in analyzed_precedents if p.get("decision") != top_outcome]

        reasoning_lines = []
        reasoning_lines.append(
            f"The predicted outcome '{OUTCOME_DISPLAY_NAMES.get(top_outcome, top_outcome)}' is supported by "
            f"{len(top_cases)} of {len(precedents)} retrieved historical precedent(s) "
            f"with an aggregate weighted evidentiary score of {top_score:.3f} ({top_support_ratio * 100:.1f}% weighted support)."
        )

        if top_cases:
            lead_case = top_cases[0]
            reasoning_lines.append(
                f"The highest-weighted precedent supporting this finding is '{lead_case.get('case_title')}' "
                f"({lead_case.get('court')}, L2 Distance: {lead_case.get('distance'):.4f}), "
                f"where the court ruled: \"{lead_case.get('decision_evidence', '')[:140]}...\""
            )

        if opposing_cases:
            opp_case = opposing_cases[0]
            reasoning_lines.append(
                f"Note: Precedent '{opp_case.get('case_title')}' ({opp_case.get('court')}, L2 Distance: {opp_case.get('distance'):.4f}) "
                f"resulted in '{OUTCOME_DISPLAY_NAMES.get(opp_case.get('decision'), opp_case.get('decision'))}', "
                f"reflecting potential judicial variance based on specific factual proof and tribunal findings."
            )

        return {
            "predicted_outcome": predicted_outcome,
            "predicted_outcome_display": OUTCOME_DISPLAY_NAMES.get(predicted_outcome, predicted_outcome),
            "evidence_strength": strength,
            "top_score": round(top_score, 3),
            "top_support_ratio": round(top_support_ratio, 3),
            "has_strong_conflict": has_strong_conflict,
            "outcome_scores": {k: round(v, 3) for k, v in sorted_outcomes},
            "outcome_distribution": dict(outcome_case_counts),
            "analyzed_precedents": analyzed_precedents,
            "synthesis_reasoning": " ".join(reasoning_lines)
        }

    def predict(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Execute full multi-case reasoning and prediction workflow.
        Returns a complete structured analysis dictionary.
        """
        # Step 1: Feature Extraction on New Case
        new_features = self.extract_features_from_text(query_text)

        # Step 2: Semantic Retrieval via FAISS
        retrieved_precedents = self.retriever.search(query_text=query_text, top_k=top_k)

        # Step 3: Evidentiary Synthesis & Outcome Analysis
        synthesis = self.synthesize_outcome(new_features, retrieved_precedents)

        # Compile Consolidated Similarities and Differences
        all_similarities = []
        all_differences = []
        for p in synthesis.get("analyzed_precedents", []):
            comp = p.get("comparison", {})
            for s in comp.get("similarities", []):
                if s not in all_similarities:
                    all_similarities.append(s)
            for d in comp.get("differences", []):
                if d not in all_differences:
                    all_differences.append(d)

        # Statutory citations list
        cited_statutes_summary = list(new_features["statutes"])
        if not cited_statutes_summary:
            cited_statutes_summary = ["Motor Vehicles Act, 1988 (General Compensation Claim Provisions)"]

        result = {
            "query_summary": query_text.strip(),
            "new_case_features": {
                "facts": list(new_features["facts"]),
                "statutes": list(new_features["statutes"])
            },
            "predicted_outcome": synthesis["predicted_outcome"],
            "predicted_outcome_display": synthesis["predicted_outcome_display"],
            "evidence_strength": synthesis["evidence_strength"],
            "outcome_distribution": synthesis["outcome_distribution"],
            "outcome_scores": synthesis["outcome_scores"],
            "synthesis_reasoning": synthesis["synthesis_reasoning"],
            "legal_similarities": all_similarities[:6],
            "important_differences": all_differences[:5],
            "applicable_statutes": cited_statutes_summary,
            "retrieved_precedents": synthesis.get("analyzed_precedents", []),
            "limitations": (
                "This prediction is generated strictly from historical precedents currently indexed "
                "in the validated prediction knowledge base. The current proof-of-concept dataset "
                "is small (7 validated cases) and designed for pipeline verification rather than formal statistical accuracy."
            ),
            "legal_disclaimer": (
                "DISCLAIMER: This system provides AI-assisted legal research and outcome prediction based on "
                "historical judicial data. It does NOT provide legal advice and does NOT guarantee what a court "
                "will decide. Actual judicial determinations depend upon full facts, testimony, evidence, "
                "statutory interpretation, procedural history, and judicial discretion."
            )
        }

        return result

    @staticmethod
    def format_prediction_report(res: Dict[str, Any]) -> str:
        """Format prediction results into a clear, professional terminal report."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("        AI-POWERED LEGAL OUTCOME ANALYSIS & CASE PREDICTION")
        lines.append("        Domain: Motor Accident Claims & Judgments (Indian Kanoon)")
        lines.append("=" * 70)

        lines.append("\n[NEW CASE SUMMARY]")
        lines.append(f"\"{res['query_summary']}\"")

        lines.append("\n" + "-" * 70)
        lines.append(f"PREDICTED OUTCOME  : {res['predicted_outcome_display'].upper()}")
        lines.append(f"EVIDENCE STRENGTH  : {res['evidence_strength']}")
        lines.append("-" * 70)

        lines.append("\n[HISTORICAL PRECEDENT EVIDENCE]")
        for idx, p in enumerate(res.get("retrieved_precedents", []), start=1):
            lines.append(f"\n{idx}. {p.get('case_title')}")
            lines.append(f"   Court               : {p.get('court')}")
            lines.append(f"   Judgment Date       : {p.get('judgment_date')}")
            
            legal_sections = p.get("legal_sections", {})
            if legal_sections:
                formatted_laws = []
                for act, sections in legal_sections.items():
                    filtered_sections = [s for s in sections if s != act]
                    if filtered_sections:
                        formatted_laws.append(f"{act} ({', '.join(filtered_sections)})")
                    else:
                        formatted_laws.append(act)
                lines.append(f"   Applicable Law      : {'; '.join(formatted_laws)}")
            else:
                lines.append(f"   Applicable Law      : Not explicitly cited in judgment text")
                
            lines.append(f"   Similarity Distance : {p.get('distance'):.4f} L2 (Lower = More Similar)")
            lines.append(f"   Historical Outcome  : {OUTCOME_DISPLAY_NAMES.get(p.get('decision'), p.get('decision'))}")
            if p.get("decision_evidence"):
                ev_clean = p.get("decision_evidence", "").replace("\n", " ").strip()
                if len(ev_clean) > 180:
                    ev_clean = ev_clean[:177] + "..."
                lines.append(f"   Operative Snippet   : \"{ev_clean}\"")

        lines.append("\n" + "-" * 70)
        lines.append("[OUTCOME ANALYSIS & DISTRIBUTION]")
        dist = res.get("outcome_distribution", {})
        if dist:
            for outcome_key, count in dist.items():
                name = OUTCOME_DISPLAY_NAMES.get(outcome_key, outcome_key)
                score = res.get("outcome_scores", {}).get(outcome_key, 0.0)
                lines.append(f"   * {name:<26} : {count} case(s) (Weighted Score: {score})")
        else:
            lines.append("   No verified historical outcomes available in retrieved records.")

        lines.append("\n" + "-" * 70)
        lines.append("[LEGAL & FACTUAL SIMILARITIES]")
        for s in res.get("legal_similarities", []):
            lines.append(f"   [+] {s}")

        lines.append("\n" + "-" * 70)
        lines.append("[IMPORTANT DIFFERENCES & NUANCES]")
        for d in res.get("important_differences", []):
            lines.append(f"   [!] {d}")

        lines.append("\n" + "-" * 70)
        lines.append("[REASONING BEHIND PREDICTION]")
        lines.append(f"{res.get('synthesis_reasoning')}")

        lines.append("\n" + "-" * 70)
        lines.append("[SYSTEM LIMITATIONS]")
        lines.append(f"{res.get('limitations')}")

        lines.append("\n" + "=" * 70)
        lines.append("[MANDATORY LEGAL DISCLAIMER]")
        lines.append(f"{res.get('legal_disclaimer')}")
        lines.append("=" * 70 + "\n")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Assisted Legal Case Outcome Prediction (Motor Accident Domain)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Facts, injury details, or legal issues of the new case"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of historical precedents to analyze (default: 3)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full analysis as structured JSON"
    )

    args = parser.parse_args()

    predictor = LegalCasePredictor()

    query_text = args.query
    if not query_text:
        print("\n" + "=" * 65)
        print("AI-ASSISTED LEGAL CASE OUTCOME PREDICTION SYSTEM")
        print("=" * 65)
        query_text = input("\nEnter facts or legal issues of the new case:\n").strip()

    if not query_text:
        print("[WARN] No case query entered. Exiting.")
        return

    result = predictor.predict(query_text=query_text, top_k=args.top_k)

    if args.json:
        # Convert any sets or non-serializable objects cleanly
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        print(predictor.format_prediction_report(result))


if __name__ == "__main__":
    main()
"""LLM-based evaluation of explanation readability."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

from recsys_nle.nl_explanations.evaluation.base import BaseEvaluator, EvaluationResult, normalise_score

# JSON detail keys returned by readability evaluation, ordered A-F to match `build_prompt` rubrics.
READABILITY_SUBSCORE_KEYS: tuple[str, ...] = (
    "fluency",
    "grammar",
    "length",
    "illustrativeness",
    "naturalness",
    "specificity",
)
_READABILITY_KEYS = READABILITY_SUBSCORE_KEYS
_READABILITY_SCORE_COUNT = len(READABILITY_SUBSCORE_KEYS)

if TYPE_CHECKING:
    import pandas as pd


def extract_subscore(evaluation: EvaluationResult | None, key: str) -> float:
    """Return a clamped readability subscore from evaluation details, or NaN."""
    if evaluation is None:
        return float("nan")
    details = evaluation.details or {}
    return normalise_score(details.get(key))


class ReadabilityEvaluator(BaseEvaluator):
    """Evaluate explanation readability using LLM scoring."""

    def build_prompt(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for evaluating explanation readability."""
        del interactions
        system_prompt = (
            "You are evaluating natural-language explanations for recommendation lists. "
            "Judge the readability of the given explanation using explicit scoring guidelines."
        )
        user_prompt = (
            "You will be provided with a candidate explanation. "
            "Evaluate it on the six criteria below, using the scoring guidelines. "
            "The explanation is intended to complete the phrase "
            '"Because you watched ..." and must be concise.\n\n'
            "Candidate explanation:\n"
            f"{explanation.strip()}\n\n"
            "Criteria and scoring guidelines (scores must be one of 1.0, 0.66, 0.33, 0.0):\n"
            "A) Fluency\n"
            '   Phrasing must be correct and fluent. The text must be appendable to "Because you watched ...".\n'
            '   Do not use bare "films before YEAR" or "films after YEAR" phrasing.\n'
            '   Use fluent forms like "films released before YEAR" or "films released after YEAR".\n'
            "- 1.0: Directly appendable, perfectly fluent\n"
            '  - "westerns and dramas from 1955 to 1995"\n'
            '  - "drama films released before 2000"\n'
            '  - "thriller films released after 2000"\n'
            "- 0.66: Appendable with minor rephrasing, or minor fluency issues\n"
            '  - "action, adventure from 1993 to 1997"\n'
            '  - "older films, comedy or horror"\n'
            '  - "comedy, drama films"\n'
            '- 0.33: Appendable with major rephrasing, or bare "film before" or "film after"\n'
            '  - "late 1980s-1990s, comedy genre"\n'
            '  - "horror films before 2000"\n'
            '  - "drama or thriller films after 2000"\n'
            "- 0.0: Cannot complete the sentence\n"
            "B) Grammar\n"
            "   The text must be grammatically correct and without typos.\n"
            "- 1.0: Perfectly grammatically correct\n"
            "- 0.66: Minor grammatical or punctuation errors or typos\n"
            "- 0.33: Serious grammatical or punctuation errors or typos (e.g. missing noun)\n"
            '  - "films from 1994, 1995 with documentary or thriller"\n'
            '  - "older films spanning action, sci-fi, comedy, or documentary"\n'
            "- 0.0: Incomprehensible meaning, or not English language\n"
            "C) Length\n"
            "   The text must be 8 words long at most.\n"
            "- 1.0: up to 8 words\n"
            "- 0.66: 9 words\n"
            '  - "older or action/sci-fi/comedy films, often 1950s-1990s"\n'
            "- 0.33: 10-11 words\n"
            "  - older or late 1930s/1940s/1990s drama, thriller, war films\n"
            "- 0.0: >11 words\n"
            "D) Illustrativeness\n"
            "   The text must not rely on negation "
            '(i.e. "not", "except", "but", etc. must not be present) or hedging '
            '(i.e. "often", "typically", "mostly", etc. must not be present).\n'
            "- 1.0: No negation or hedging\n"
            '  - "drama films from pre-1990s"\n'
            '  - "horror/sci-fi or action/crime/drama from 1995 to 1998"\n'
            "- 0.66: Minor negative framing\n"
            '  - "romance films from 1998 excluding comedy"\n'
            "- 0.33: Mix of positive and negative framing or partly hedged\n"
            '  - (e.g. "films from 1995 excluding comedies")\n'
            '  - "comedy films, often 1950s-1990s"\n'
            "- 0.0: Primarily negative framing or hedged\n"
            '  - "films except of comedies"\n'
            '  - "mostly comedies"\n'
            "E) Naturalness\n"
            "   The explanation must not reveal or imply properties of the underlying dataset "
            "(e.g., missing values, feature availability, value counts, or schema artifacts).\n"
            "   It should feel natural and user-facing, not analytical or data-driven.\n"
            "- 1.0: No dataset references; fully natural phrasing\n"
            '  - "action and drama films from the 1990s"\n'
            "- 0.66: Slightly artificial phrasing hinting at aggregation or structure\n"
            '  - "films spanning multiple genres from the 1990s"\n'
            "- 0.33: Clear indirect reference to dataset properties (counts, grouping, completeness)\n"
            '  - "films featuring multiple genres"\n'
            '  - "single-genre films from the late 1980s-1990s"\n'
            "- 0.0: Explicit reference to dataset artifacts (missing data, fields, statistics)\n"
            '  - "films with missing release years"\n'
            '  - "documentaries with missing genre information"\n'
            '  - "films with multiple genres present"\n'
            "F) Specificity\n"
            "   The explanation must not be overly broad in time period or genre coverage.\n"
            "   Time spans: a continuous range or adjacent decades are acceptable only if the "
            "total span is at most 30 years.\n"
            "   Time periods: at most two time periods are acceptable; if more than two are "
            'mentioned, they must be adjacent (e.g. "1970s, 1980s and 1990s").\n'
            "   Genres: at most three distinct genre categories may be mentioned.\n"
            "- 1.0: Narrow and precise on all dimensions\n"
            '  - "drama films from 1980s-1990s"\n'
            '  - "action and thriller films from 1995 to 1998"\n'
            "- 0.66: Slightly broad but within all limits\n"
            '  - "westerns and dramas from 1965 to 1995"\n'
            '  - "dramas from the 1970s or 1990s"\n'
            "- 0.33: Clearly too broad on one dimension\n"
            '  - "dramas or comedies, 1940s-1990s"\n'
            '  - "dramas released in 1980, 1994, or 1998"\n'
            '  - "action, drama, thriller, horror films from 1984-1999"\n'
            "- 0.0: Very broad across multiple dimensions\n"
            '  - "action, drama, thriller, and horror films from the 1940s through the 2000s"\n'
            "Respond ONLY with a JSON object of the form:\n"
            '{"fluency": <score>, "grammar": <score>, "length": <score>, '
            '"illustrativeness": <score>, "naturalness": <score>, "specificity": <score>}.\n'
            "Do not include any extra keys, text, or commentary outside this JSON object."
        )
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def parse_result(self, raw_output: str, *, prompt: str | None = None) -> EvaluationResult:
        """Parse raw LLM output into a structured readability evaluation result."""
        parsed, parse_error = self._normalise_payload(raw_output)
        issues: list[str] = []
        if parse_error:
            issues.append("LLM output for readability could not be parsed as JSON; scores set to NaN.")

        scores = self._extract_scores(parsed, issues)
        overall = self._compute_overall_score(scores)
        judgment = self._build_judgment(scores, overall)
        details: dict[str, object] = {**scores, "overall": overall}
        if issues:
            details["warnings"] = issues
        return EvaluationResult(judgment=judgment, score=overall, details=details, prompt=prompt)

    @staticmethod
    def _extract_scores(payload: Mapping[str, object], issues: list[str]) -> dict[str, float]:
        """Extract readability subscores from payload."""
        scores: dict[str, float] = {}
        for key in _READABILITY_KEYS:
            if key not in payload:
                issues.append(f"LLM output missing '{key}' readability score.")
                scores[key] = float("nan")
                continue
            scores[key] = normalise_score(payload.get(key))
        return scores

    @staticmethod
    def _compute_overall_score(scores: Mapping[str, float]) -> float:
        """Compute overall readability mean score or NaN."""
        values = [value for value in scores.values() if isinstance(value, (int, float))]
        if len(values) < _READABILITY_SCORE_COUNT or any(math.isnan(value) for value in values):
            return float("nan")
        return float(sum(values) / len(values))

    @staticmethod
    def _build_judgment(scores: Mapping[str, float], overall: float) -> str:
        """Build a concise judgment summary for readability results."""
        if math.isnan(overall):
            return "Readability scores could not be computed from the LLM output."
        return (
            "Readability scores - fluency: "
            f"{scores.get('fluency', float('nan')):.2f}, "
            "grammar: "
            f"{scores.get('grammar', float('nan')):.2f}, "
            "length: "
            f"{scores.get('length', float('nan')):.2f}, "
            "illustrativeness: "
            f"{scores.get('illustrativeness', float('nan')):.2f}, "
            "naturalness: "
            f"{scores.get('naturalness', float('nan')):.2f}, "
            "specificity: "
            f"{scores.get('specificity', float('nan')):.2f}."
        )

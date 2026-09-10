import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from openai import OpenAI
from config.settings import settings


@dataclass
class ContradictionPair:
    statement1: str
    statement2: str
    explanation: str
    severity: str


@dataclass
class HallucinationCheck:
    claim: str
    is_supported: bool
    supporting_chunks: list[str]
    confidence: float
    explanation: str


@dataclass
class CompletenessAnalysis:
    query_aspects: list[str]
    addressed_aspects: list[str]
    missing_aspects: list[str]
    completeness_score: float
    explanation: str


@dataclass
class CitationValidation:
    total_citations: int
    valid_citations: int
    invalid_citations: list[str]
    uncited_claims: list[str]
    citation_coverage: float


@dataclass
class VerificationResult:
    query: str
    answer: str
    source_chunks: list[dict[str, Any]]
    completeness: CompletenessAnalysis
    contradictions: list[ContradictionPair]
    hallucinations: list[HallucinationCheck]
    citation_validation: CitationValidation
    overall_score: float
    is_trustworthy: bool
    issues_found: list[str]
    recommendations: list[str]
    verification_timestamp: datetime = field(default_factory=datetime.now)
    verification_cost: float = 0.0


class AnswerVerifier:

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.cost_per_1k_input = 0.0025
        self.cost_per_1k_output = 0.01
        self.min_completeness_score = 70.0
        self.min_citation_coverage = 80.0
        self.max_hallucination_rate = 0.1

    def verify_answer(
        self,
        query: str,
        answer: str,
        source_chunks: list[dict[str, Any]],
        check_completeness: bool = True,
        check_contradictions: bool = True,
        check_hallucinations: bool = True,
        check_citations: bool = True,
    ) -> VerificationResult:
        total_cost = 0.0
        if check_completeness:
            completeness, cost = self._check_completeness(query, answer)
            total_cost += cost
        else:
            completeness = CompletenessAnalysis(
                query_aspects=[],
                addressed_aspects=[],
                missing_aspects=[],
                completeness_score=100.0,
                explanation="Completeness check skipped",
            )
        if check_contradictions:
            contradictions, cost = self._detect_contradictions(answer)
            total_cost += cost
        else:
            contradictions = []
        if check_hallucinations:
            hallucinations, cost = self._detect_hallucinations(answer, source_chunks)
            total_cost += cost
        else:
            hallucinations = []
        if check_citations:
            citation_validation = self._validate_citations(answer, source_chunks)
        else:
            citation_validation = CitationValidation(
                total_citations=0,
                valid_citations=0,
                invalid_citations=[],
                uncited_claims=[],
                citation_coverage=100.0,
            )
        overall_score, is_trustworthy, issues, recommendations = (
            self._calculate_overall_assessment(
                completeness, contradictions, hallucinations, citation_validation
            )
        )
        return VerificationResult(
            query=query,
            answer=answer,
            source_chunks=source_chunks,
            completeness=completeness,
            contradictions=contradictions,
            hallucinations=hallucinations,
            citation_validation=citation_validation,
            overall_score=overall_score,
            is_trustworthy=is_trustworthy,
            issues_found=issues,
            recommendations=recommendations,
            verification_cost=total_cost,
        )

    def _check_completeness(
        self, query: str, answer: str
    ) -> tuple[CompletenessAnalysis, float]:
        prompt = f'You are an expert at analyzing whether answers fully address queries.\n\nQUERY: {query}\n\nANSWER: {answer}\n\nYour task:\n1. Break down the query into all aspects/sub-questions it asks\n2. Check which aspects are addressed in the answer\n3. Identify any missing aspects\n4. Calculate a completeness score (0-100%)\n\nRespond in this EXACT JSON format:\n{{\n"query_aspects": ["aspect 1", "aspect 2", ...],\n"addressed_aspects": ["aspect 1", ...],\n"missing_aspects": ["aspect 2", ...],\n"completeness_score": 85.0,\n"explanation": "Brief explanation of completeness"\n}}'
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert answer quality analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        cost = self._calculate_cost(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        import json

        data = json.loads(response.choices[0].message.content)
        return (
            CompletenessAnalysis(
                query_aspects=data["query_aspects"],
                addressed_aspects=data["addressed_aspects"],
                missing_aspects=data["missing_aspects"],
                completeness_score=data["completeness_score"],
                explanation=data["explanation"],
            ),
            cost,
        )

    def _detect_contradictions(
        self, answer: str
    ) -> tuple[list[ContradictionPair], float]:
        prompt = f'You are an expert at detecting contradictions in text.\n\nANSWER TO ANALYZE:\n{answer}\n\nYour task:\nFind any contradictory statements in this answer. A contradiction occurs when two statements \ncannot both be true at the same time.\n\nRespond in this EXACT JSON format:\n{{\n"contradictions": [\n{{\n"statement1": "First contradictory statement",\n"statement2": "Second contradictory statement",\n"explanation": "Why these contradict",\n"severity": "high/medium/low"\n}}\n]\n}}\n\nIf no contradictions found, return: {{"contradictions": []}}'
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert logical consistency analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        cost = self._calculate_cost(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        import json

        data = json.loads(response.choices[0].message.content)
        contradictions = [
            ContradictionPair(
                statement1=c["statement1"],
                statement2=c["statement2"],
                explanation=c["explanation"],
                severity=c["severity"],
            )
            for c in data.get("contradictions", [])
        ]
        return (contradictions, cost)

    def _detect_hallucinations(
        self, answer: str, source_chunks: list[dict[str, Any]]
    ) -> tuple[list[HallucinationCheck], float]:
        source_text = "\n\n".join(
            [
                f"[SOURCE {i + 1}]: {(chunk.text if hasattr(chunk, 'text') else chunk['text'])}"
                for i, chunk in enumerate(source_chunks)
            ]
        )
        prompt = f"""You are an expert at detecting hallucinations in AI-generated answers.\n\nSOURCE DOCUMENTS:\n{source_text}\n\nGENERATED ANSWER:\n{answer}\n\nYour task:\n1. Extract all factual claims from the answer\n2. For each claim, check if it's supported by the source documents\n3. Mark claims as supported or unsupported (hallucination)\n\nRespond in this EXACT JSON format:\n{{\n"hallucination_checks": [\n{{\n"claim": "The factual claim",\n"is_supported": true/false,\n"supporting_sources": ["SOURCE 1", "SOURCE 2"],\n"confidence": 0.95,\n"explanation": "Why supported or not"\n}}\n]\n}}"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert fact-checking analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        cost = self._calculate_cost(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        import json

        data = json.loads(response.choices[0].message.content)
        hallucinations = [
            HallucinationCheck(
                claim=h["claim"],
                is_supported=h["is_supported"],
                supporting_chunks=h.get("supporting_sources", []),
                confidence=h["confidence"],
                explanation=h["explanation"],
            )
            for h in data.get("hallucination_checks", [])
        ]
        return (hallucinations, cost)

    def _validate_citations(
        self, answer: str, source_chunks: list[dict[str, Any]]
    ) -> CitationValidation:
        citation_pattern = "\\[([^\\]]+)\\]"
        citations = re.findall(citation_pattern, answer)
        valid_chunk_ids = {
            chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id", "")
            for chunk in source_chunks
        }
        valid_citations = []
        invalid_citations = []
        for citation in citations:
            if citation in valid_chunk_ids:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        total_citations = len(citations)
        valid_count = len(valid_citations)
        citation_coverage = (
            valid_count / total_citations * 100 if total_citations > 0 else 0.0
        )
        sentences = re.split("[.!?]+", answer)
        uncited_claims = [
            sent.strip()
            for sent in sentences
            if sent.strip() and (not re.search(citation_pattern, sent))
        ]
        return CitationValidation(
            total_citations=total_citations,
            valid_citations=valid_count,
            invalid_citations=invalid_citations,
            uncited_claims=uncited_claims[:3],
            citation_coverage=citation_coverage,
        )

    def _calculate_overall_assessment(
        self,
        completeness: CompletenessAnalysis,
        contradictions: list[ContradictionPair],
        hallucinations: list[HallucinationCheck],
        citation_validation: CitationValidation,
    ) -> tuple[float, bool, list[str], list[str]]:
        issues = []
        recommendations = []
        score = 100.0
        if completeness.completeness_score < 100:
            deduction = (100 - completeness.completeness_score) * 0.3
            score -= deduction
            if completeness.missing_aspects:
                issues.append(
                    f"Missing aspects: {', '.join(completeness.missing_aspects)}"
                )
                recommendations.append("Add information about missing query aspects")
        if contradictions:
            high_severity = sum((1 for c in contradictions if c.severity == "high"))
            medium_severity = sum((1 for c in contradictions if c.severity == "medium"))
            low_severity = sum((1 for c in contradictions if c.severity == "low"))
            deduction = high_severity * 20 + medium_severity * 10 + low_severity * 5
            score -= deduction
            issues.append(f"Found {len(contradictions)} contradiction(s)")
            recommendations.append("Resolve contradictory statements")
        if hallucinations:
            unsupported = [h for h in hallucinations if not h.is_supported]
            if unsupported:
                hallucination_rate = len(unsupported) / len(hallucinations)
                deduction = hallucination_rate * 30
                score -= deduction
                issues.append(f"Found {len(unsupported)} unsupported claim(s)")
                recommendations.append("Remove or verify unsupported claims")
        if citation_validation.citation_coverage < 100:
            deduction = (100 - citation_validation.citation_coverage) * 0.2
            score -= deduction
            if citation_validation.invalid_citations:
                issues.append(
                    f"Invalid citations: {len(citation_validation.invalid_citations)}"
                )
                recommendations.append("Fix invalid citation references")
            if citation_validation.uncited_claims:
                issues.append(
                    f"Uncited claims: {len(citation_validation.uncited_claims)}"
                )
                recommendations.append("Add citations to support claims")
        score = max(0.0, min(100.0, score))
        is_trustworthy = (
            score >= 70.0
            and completeness.completeness_score >= self.min_completeness_score
            and (len([c for c in contradictions if c.severity == "high"]) == 0)
            and (citation_validation.citation_coverage >= self.min_citation_coverage)
        )
        if not issues:
            issues.append("No major issues found")
        if not recommendations:
            recommendations.append("Answer meets quality standards")
        return (score, is_trustworthy, issues, recommendations)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = input_tokens / 1000 * self.cost_per_1k_input
        output_cost = output_tokens / 1000 * self.cost_per_1k_output
        return input_cost + output_cost

    def format_verification_report(self, result: VerificationResult) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("ANSWER VERIFICATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nOVERALL SCORE: {result.overall_score:.1f}%")
        trustworthy_emoji = "" if result.is_trustworthy else ""
        lines.append(
            f"{trustworthy_emoji} TRUSTWORTHY: {('YES' if result.is_trustworthy else 'NO')}"
        )
        lines.append("\nQUERY:")
        lines.append(f"{result.query}")
        lines.append("\nANSWER:")
        lines.append(f"{result.answer[:200]}...")
        lines.append(f"\nCOMPLETENESS: {result.completeness.completeness_score:.1f}%")
        if result.completeness.missing_aspects:
            lines.append(
                f"Missing: {', '.join(result.completeness.missing_aspects)}"
            )
        if result.contradictions:
            lines.append(f"\nCONTRADICTIONS FOUND: {len(result.contradictions)}")
            for i, contra in enumerate(result.contradictions, 1):
                lines.append(f"[{i}] {contra.severity.upper()}")
                lines.append(f"Statement 1: {contra.statement1[:80]}...")
                lines.append(f"Statement 2: {contra.statement2[:80]}...")
        else:
            lines.append("\nCONTRADICTIONS: None detected")
        unsupported = [h for h in result.hallucinations if not h.is_supported]
        if unsupported:
            lines.append(f"\nHALLUCINATIONS FOUND: {len(unsupported)}")
            for i, hall in enumerate(unsupported[:3], 1):
                lines.append(f"[{i}] {hall.claim[:80]}...")
                lines.append(f"Confidence: {hall.confidence:.2f}")
        else:
            lines.append("\nHALLUCINATIONS: None detected")
        lines.append("\nCITATIONS:")
        lines.append(f"Total: {result.citation_validation.total_citations}")
        lines.append(f"Valid: {result.citation_validation.valid_citations}")
        lines.append(
            f"Coverage: {result.citation_validation.citation_coverage:.1f}%"
        )
        lines.append("\nISSUES:")
        for issue in result.issues_found:
            lines.append(f"• {issue}")
        lines.append("\nRECOMMENDATIONS:")
        for rec in result.recommendations:
            lines.append(f"• {rec}")
        lines.append("\nMETADATA:")
        lines.append(f"Sources used: {len(result.source_chunks)}")
        lines.append(f"Verification cost: ${result.verification_cost:.4f}")
        lines.append(
            f"Timestamp: {result.verification_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


def quick_verify(
    query: str, answer: str, source_chunks: list[dict[str, Any]]
) -> VerificationResult:
    verifier = AnswerVerifier()
    return verifier.verify_answer(query, answer, source_chunks)

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


@dataclass
class Citation:
    chunk_id: str
    contract_id: str
    text_snippet: str
    claim: str


@dataclass
class Critique:
    completeness_score: int
    accuracy_score: int
    citation_quality: int
    overall_quality: int
    issues_found: list[str]
    suggestions: list[str]
    needs_refinement: bool


@dataclass
class AnswerIteration:
    iteration_num: int
    answer: str
    critique: Critique
    tokens_used: int
    cost_usd: float


@dataclass
class SelfCorrectionResult:
    query: str
    final_answer: str
    citations: list[Citation]
    confidence: int
    num_iterations: int
    iterations: list[AnswerIteration]
    total_tokens: int
    total_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


class SelfCorrectionAgent:
    PRICE_PER_1K_INPUT_TOKENS = 0.01
    PRICE_PER_1K_OUTPUT_TOKENS = 0.03

    def __init__(
        self, model: str = "gpt-4", max_iterations: int = 3, quality_threshold: int = 85
    ):
        print("Initializing Self-Correction Agent...")
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        print("OpenAI client ready")
        print(f"Model: {model}")
        print(f"Max iterations: {max_iterations}")
        print(f"Quality threshold: {quality_threshold}%")

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict],
        max_iterations: int | None = None,
        quality_threshold: int | None = None,
    ) -> SelfCorrectionResult:
        max_iter = max_iterations or self.max_iterations
        quality_thresh = quality_threshold or self.quality_threshold
        print("\nGenerating answer with self-correction...")
        print(f"Query: '{query}'")
        print(f"Retrieved chunks: {len(retrieved_chunks)}")
        iterations = []
        total_tokens = 0
        total_cost = 0.0
        context = self._format_context(retrieved_chunks)
        print("\nIteration 1: Generating initial answer...")
        answer = self._generate_initial_answer(query, context)
        tokens_used, cost = self._estimate_tokens_and_cost(context, answer)
        total_tokens += tokens_used
        total_cost += cost
        print("Critiquing answer...")
        critique = self._critique_answer(query, answer, context)
        iterations.append(
            AnswerIteration(
                iteration_num=1,
                answer=answer,
                critique=critique,
                tokens_used=tokens_used,
                cost_usd=cost,
            )
        )
        print(f"Quality: {critique.overall_quality}%")
        print(f"Completeness: {critique.completeness_score}%")
        print(f"Accuracy: {critique.accuracy_score}%")
        current_answer = answer
        iteration_num = 1
        while iteration_num < max_iter and critique.needs_refinement:
            if critique.overall_quality >= quality_thresh:
                print(
                    f"Quality threshold met ({critique.overall_quality}% >= {quality_thresh}%)"
                )
                break
            iteration_num += 1
            print(f"\nIteration {iteration_num}: Refining answer...")
            refined_answer = self._refine_answer(
                query, current_answer, critique, context
            )
            tokens_used, cost = self._estimate_tokens_and_cost(context, refined_answer)
            total_tokens += tokens_used
            total_cost += cost
            print("Critiquing refined answer...")
            critique = self._critique_answer(query, refined_answer, context)
            iterations.append(
                AnswerIteration(
                    iteration_num=iteration_num,
                    answer=refined_answer,
                    critique=critique,
                    tokens_used=tokens_used,
                    cost_usd=cost,
                )
            )
            print(f"Quality: {critique.overall_quality}%")
            print(
                f"Improvement: {critique.overall_quality - iterations[-2].critique.overall_quality:+d}%"
            )
            current_answer = refined_answer
        citations = self._extract_citations(current_answer, retrieved_chunks)
        result = SelfCorrectionResult(
            query=query,
            final_answer=current_answer,
            citations=citations,
            confidence=critique.overall_quality,
            num_iterations=iteration_num,
            iterations=iterations,
            total_tokens=total_tokens,
            total_cost=total_cost,
        )
        print("\nAnswer generation complete!")
        print(f"Final confidence: {result.confidence}%")
        print(f"Iterations: {result.num_iterations}")
        print(f"Citations: {len(result.citations)}")
        print(f"Total cost: ${result.total_cost:.4f}")
        return result

    def _format_context(self, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            if hasattr(chunk, "to_dict"):
                chunk = chunk.to_dict()
            context_parts.append(
                f"[Source {i}] Contract: {chunk.get('contract_id', 'Unknown')}\n{chunk.get('text', '')}\n"
            )
        return "\n".join(context_parts)

    def _generate_initial_answer(self, query: str, context: str) -> str:
        system_prompt = 'You are a legal contract analysis expert. Your task is to answer questions based ONLY on the provided contract excerpts.\n\nRules:\n1. Answer must be based ONLY on provided sources\n2. Cite sources for every claim using [Source N] notation\n3. If information is not in sources, say "Information not found in provided contracts"\n4. Be precise and specific\n5. Use exact quotes when possible\n6. Structure answer clearly with sections if needed'
        user_prompt = f"Question: {query}\n\nContract Sources:\n{context}\n\nPlease provide a comprehensive answer based on the sources above. Remember to cite sources for every claim."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def _critique_answer(self, query: str, answer: str, context: str) -> Critique:
        critique_prompt = f'You are a quality assurance expert. Evaluate this answer for quality.\n\nOriginal Question: {query}\n\nAnswer to Evaluate:\n{answer}\n\nAvailable Context:\n{context}\n\nEvaluate on these criteria (score each 0-100):\n1. Completeness: Does it fully answer the question?\n2. Accuracy: Are all claims supported by sources?\n3. Citation Quality: Are sources properly cited?\n4. Overall Quality: General assessment\n\nRespond in JSON format:\n{{\n"completeness_score": <0-100>,\n"accuracy_score": <0-100>,\n"citation_quality": <0-100>,\n"overall_quality": <0-100>,\n"issues_found": ["issue 1", "issue 2"],\n"suggestions": ["suggestion 1", "suggestion 2"],\n"needs_refinement": true/false\n}}'
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a quality assurance expert. Respond only with valid JSON.",
                },
                {"role": "user", "content": critique_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        critique_json = json.loads(response.choices[0].message.content)
        return Critique(
            completeness_score=critique_json["completeness_score"],
            accuracy_score=critique_json["accuracy_score"],
            citation_quality=critique_json["citation_quality"],
            overall_quality=critique_json["overall_quality"],
            issues_found=critique_json["issues_found"],
            suggestions=critique_json["suggestions"],
            needs_refinement=critique_json["needs_refinement"],
        )

    def _refine_answer(
        self, query: str, current_answer: str, critique: Critique, context: str
    ) -> str:
        refinement_prompt = f"You are improving a previous answer based on feedback.\n\nOriginal Question: {query}\n\nCurrent Answer:\n{current_answer}\n\nQuality Scores:\n- Completeness: {critique.completeness_score}%\n- Accuracy: {critique.accuracy_score}%\n- Citation Quality: {critique.citation_quality}%\n\nIssues Found:\n{chr(10).join((f'- {issue}' for issue in critique.issues_found))}\n\nSuggestions for Improvement:\n{chr(10).join((f'- {suggestion}' for suggestion in critique.suggestions))}\n\nAvailable Context:\n{context}\n\nPlease provide an improved answer that addresses the issues and follows the suggestions. Maintain proper citations."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal expert improving your previous answer based on feedback.",
                },
                {"role": "user", "content": refinement_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def _extract_citations(self, answer: str, chunks: list[dict]) -> list[Citation]:
        citations = []
        import re

        source_refs = re.findall("\\[Source (\\d+)\\]", answer)
        for ref in set(source_refs):
            idx = int(ref) - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                if hasattr(chunk, "to_dict"):
                    chunk = chunk.to_dict()
                claim_match = re.search(f"([^.]*\\[Source {ref}\\][^.]*\\.)", answer)
                claim = claim_match.group(1) if claim_match else "See source"
                citations.append(
                    Citation(
                        chunk_id=chunk.get("chunk_id", "unknown"),
                        contract_id=chunk.get("contract_id", "unknown"),
                        text_snippet=chunk.get("text", "")[:200],
                        claim=claim.strip(),
                    )
                )
        return citations

    def _estimate_tokens_and_cost(self, context: str, answer: str) -> tuple[int, float]:
        input_tokens = len(context) // 4
        output_tokens = len(answer) // 4
        total_tokens = input_tokens + output_tokens
        cost = (
            input_tokens / 1000 * self.PRICE_PER_1K_INPUT_TOKENS
            + output_tokens / 1000 * self.PRICE_PER_1K_OUTPUT_TOKENS
        )
        return (total_tokens, cost)


def format_result(result: SelfCorrectionResult) -> str:
    output = []
    output.append("\n" + "=" * 70)
    output.append("SELF-CORRECTION AGENT - RESULT")
    output.append("=" * 70)
    output.append(f"\nQuery: {result.query}")
    output.append(f"\n{'' * 70}")
    output.append("\nFINAL ANSWER:")
    output.append(result.final_answer)
    output.append(f"\n{'' * 70}")
    output.append("\nMETADATA:")
    output.append(f"Confidence: {result.confidence}%")
    output.append(f"Iterations: {result.num_iterations}")
    output.append(f"Citations: {len(result.citations)}")
    output.append(f"Total Cost: ${result.total_cost:.4f}")
    if result.citations:
        output.append(f"\n{'' * 70}")
        output.append("\nCITATIONS:")
        for i, cite in enumerate(result.citations, 1):
            output.append(f"\n[{i}] {cite.contract_id} ({cite.chunk_id})")
            output.append(f"Claim: {cite.claim}")
            output.append(f"Source: {cite.text_snippet}...")
    output.append(f"\n{'' * 70}")
    output.append("\nITERATION HISTORY:")
    for iter_data in result.iterations:
        output.append(f"\nIteration {iter_data.iteration_num}:")
        output.append(f"Quality: {iter_data.critique.overall_quality}%")
        output.append(f"Cost: ${iter_data.cost_usd:.4f}")
        if iter_data.critique.issues_found:
            output.append(f"Issues: {', '.join(iter_data.critique.issues_found[:2])}")
    output.append("\n" + "=" * 70)
    return "\n".join(output)


if __name__ == "__main__":
    print("=" * 70)
    print("Self-Correction Agent - Test")
    print("=" * 70)
    agent = SelfCorrectionAgent(model="gpt-4", max_iterations=3, quality_threshold=85)
    mock_chunks = [
        {
            "chunk_id": "cuad_0001_chunk_5",
            "contract_id": "cuad_0001",
            "text": "This Agreement may be terminated by either party upon 60 days written notice to the other party.",
            "score": 0.95,
        },
        {
            "chunk_id": "cuad_0001_chunk_6",
            "contract_id": "cuad_0001",
            "text": "Upon termination, all confidential information must be returned within 30 days.",
            "score": 0.87,
        },
    ]
    result = agent.generate_answer(
        query="What are the termination notice requirements?",
        retrieved_chunks=mock_chunks,
    )
    print(format_result(result))

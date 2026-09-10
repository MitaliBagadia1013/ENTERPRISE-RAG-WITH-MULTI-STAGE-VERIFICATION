import os
import re
from openai import OpenAI


class QueryExpander:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.legal_synonyms = {
            "termination": [
                "end",
                "cancellation",
                "exit",
                "cessation",
                "dissolution",
                "conclude",
            ],
            "renewal": ["extension", "continuation", "re-execution", "prolongation"],
            "execution": ["signing", "enactment", "effectuation", "performance"],
            "confidentiality": [
                "non-disclosure",
                "secrecy",
                "proprietary information",
                "trade secrets",
            ],
            "intellectual property": [
                "IP",
                "patents",
                "trademarks",
                "copyrights",
                "proprietary rights",
            ],
            "payment": [
                "compensation",
                "remuneration",
                "fee",
                "consideration",
                "settlement",
            ],
            "penalty": ["fine", "damages", "liquidated damages", "forfeiture"],
            "indemnification": [
                "indemnity",
                "reimbursement",
                "compensation",
                "hold harmless",
            ],
            "liability": ["responsibility", "accountability", "obligation", "duty"],
            "breach": [
                "violation",
                "infringement",
                "default",
                "non-compliance",
                "contravention",
            ],
            "warranty": ["guarantee", "assurance", "representation", "covenant"],
            "governing law": [
                "applicable law",
                "jurisdiction",
                "legal framework",
                "choice of law",
            ],
            "arbitration": [
                "dispute resolution",
                "mediation",
                "alternative dispute resolution",
                "ADR",
            ],
            "force majeure": [
                "act of god",
                "unforeseeable circumstances",
                "impossibility",
            ],
            "delivery": ["provision", "supply", "furnishing", "performance"],
            "milestone": ["deliverable", "checkpoint", "phase", "stage"],
            "deadline": ["due date", "timeline", "timeframe", "completion date"],
        }
        self.entity_patterns = [
            "\\b(termination|renewal|confidentiality|liability|indemnification|breach|warranty)\\b",
            "\\b(payment|penalty|fee|compensation)\\b",
            "\\b(deadline|milestone|delivery)\\b",
        ]

    def expand_query(self, query: str, max_terms: int = 6) -> dict[str, any]:
        entities = self._extract_entities(query)
        sub_queries = self._decompose_query(query)
        expanded = self._expand_with_synonyms(query, max_terms)
        if expanded == query and (not sub_queries):
            expanded = self._llm_expansion(query, max_terms)
            strategy = "llm_expansion"
        elif sub_queries:
            strategy = "decomposition"
        else:
            strategy = "synonym_expansion"
        return {
            "expanded_query": expanded,
            "sub_queries": sub_queries,
            "entities": entities,
            "strategy_used": strategy,
        }

    def _extract_entities(self, query: str) -> list[str]:
        entities = []
        query_lower = query.lower()
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities.extend(matches)
        for term in self.legal_synonyms.keys():
            if term in query_lower:
                entities.append(term)
        return list(set(entities))

    def _expand_with_synonyms(self, query: str, max_terms: int) -> str:
        query_lower = query.lower()
        expanded_terms = [query]
        terms_added = 0
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower and terms_added < max_terms:
                remaining_slots = max_terms - terms_added
                expanded_terms.extend(synonyms[:remaining_slots])
                terms_added += len(synonyms[:remaining_slots])
        unique_terms = []
        seen = set()
        for term in expanded_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                unique_terms.append(term)
                seen.add(term_lower)
        return "".join(unique_terms)

    def _decompose_query(self, query: str) -> list[str]:
        complex_indicators = ["and ", "or ", ",", "?"]
        if not any((ind in query.lower() for ind in complex_indicators)):
            return []
        sub_queries = []
        parts = re.split("\\s+(?:and|or)\\s+", query, flags=re.IGNORECASE)
        if len(parts) > 1:
            for part in parts:
                cleaned = part.strip().rstrip("?,.")
                if len(cleaned.split()) >= 2:
                    sub_queries.append(cleaned)
        return sub_queries

    def _llm_expansion(self, query: str, max_terms: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal contract expert. Generate 4-6 related legal terms or synonyms for the given query. Focus on contract terminology. Output ONLY the terms, comma-separated, no explanations.",
                    },
                    {"role": "user", "content": f"Query: {query}"},
                ],
                max_tokens=60,
                temperature=0.3,
            )
            expansion = response.choices[0].message.content.strip()
            terms = [t.strip() for t in expansion.split(",")[:max_terms]]
            return f"{query} {' '.join(terms)}"
        except Exception as e:
            print(f"LLM expansion failed: {e}")
            return query


if __name__ == "__main__":
    expander = QueryExpander()
    test_queries = [
        "termination conditions",
        "confidentiality obligations",
        "payment terms and delivery milestones",
        "What happens if there is a breach?",
    ]
    print("=" * 70)
    print("QUERY EXPANSION EXAMPLES")
    print("=" * 70)
    for query in test_queries:
        result = expander.expand_query(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {result['expanded_query']}")
        print(f"Strategy: {result['strategy_used']}")
        if result["entities"]:
            print(f"Entities: {', '.join(result['entities'])}")
        if result["sub_queries"]:
            print(f"Sub-queries: {result['sub_queries']}")
    print("\n" + "=" * 70)

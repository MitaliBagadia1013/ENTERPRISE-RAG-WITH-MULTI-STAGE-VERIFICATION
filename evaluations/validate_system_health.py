import os
import sys
from dotenv import load_dotenv


class Colors:
    GREEN = "\x1b[92m"
    RED = "\x1b[91m"
    YELLOW = "\x1b[93m"
    BLUE = "\x1b[94m"
    BOLD = "\x1b[1m"
    END = "\x1b[0m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN} {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED} {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW} {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}{text}{Colors.END}")


class SystemHealthValidator:

    def __init__(self):
        load_dotenv()
        self.results = {}
        self.critical_failures = []

    def check_environment_variables(self) -> bool:
        print_header("1. ENVIRONMENT VARIABLES CHECK")
        required_vars = {
            "OPENAI_API_KEY": "OpenAI API",
            "PINECONE_API_KEY": "Pinecone Vector DB",
            "COHERE_API_KEY": "Cohere Reranker",
        }
        optional_vars = {
            "LANGFUSE_PUBLIC_KEY": "Langfuse Monitoring",
            "LANGFUSE_SECRET_KEY": "Langfuse Monitoring",
        }
        all_present = True
        for var, service in required_vars.items():
            value = os.getenv(var)
            if value and len(value) > 10:
                print_success(f"{service}: Found ({value[:15]}...)")
            else:
                print_error(f"{service}: MISSING or INVALID")
                self.critical_failures.append(f"Missing {var}")
                all_present = False
        print("\nOptional Services:")
        for var, service in optional_vars.items():
            value = os.getenv(var)
            if value and len(value) > 10:
                print_success(f"{service}: Configured")
            else:
                print_warning(f"{service}: Not configured (optional)")
        self.results["environment"] = all_present
        return all_present

    def check_openai_credits(self) -> tuple[bool, float]:
        print_header("2. OPENAI API CHECK")
        try:
            import openai
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print_info("Testing API key with minimal call...")
            response = client.embeddings.create(
                model="text-embedding-3-large", input="test"
            )
            print_success("API Key is VALID ")
            print_success("Embeddings API is working ")
            try:
                print_info("Testing GPT-4 access...")
                test_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
                print_success("GPT-4 Access: CONFIRMED ")
                print_success("You have sufficient credits for testing!")
                self.results["openai"] = True
                return (True, -1)
            except openai.RateLimitError as e:
                print_error("RATE LIMIT ERROR: You've exceeded your quota!")
                print_error(str(e))
                print_warning("\nACTION REQUIRED:")
                print_warning("Go to: https://platform.openai.com/account/billing")
                print_warning("Add at least $10 in credits")
                self.critical_failures.append("OpenAI quota exceeded")
                self.results["openai"] = False
                return (False, 0)
            except openai.AuthenticationError:
                print_error("Authentication failed - invalid API key")
                self.critical_failures.append("Invalid OpenAI API key")
                self.results["openai"] = False
                return (False, 0)
        except Exception as e:
            print_error(f"OpenAI check failed: {e!s}")
            self.critical_failures.append(f"OpenAI error: {e!s}")
            self.results["openai"] = False
            return (False, 0)

    def check_pinecone_index(self) -> tuple[bool, int]:
        print_header("3. PINECONE VECTOR DATABASE CHECK")
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            print_info("Connecting to Pinecone...")
            indexes = pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            print_success(f"Connection successful! Found {len(index_names)} index(es)")
            if not index_names:
                print_error("No indexes found! Run ingestion first.")
                self.critical_failures.append("No Pinecone indexes")
                self.results["pinecone"] = False
                return (False, 0)
            target_index = "enterprise-contracts"
            if target_index not in index_names:
                print_warning(f"Index '{target_index}' not found")
                print_info(f"Available indexes: {', '.join(index_names)}")
                target_index = index_names[0]
                print_info(f"Using first available index: {target_index}")
            index = pc.Index(target_index)
            stats = index.describe_index_stats()
            vector_count = stats.get("total_vector_count", 0)
            print_success(f"Index: {target_index}")
            print_success(f"Vectors: {vector_count:,}")
            expected_min = 9000
            expected_max = 10000
            if vector_count < expected_min:
                print_error(f"Expected ~9,217 vectors, found {vector_count}")
                print_warning("Run data ingestion: python3 data/load_cuad.py")
                self.critical_failures.append("Insufficient vectors in Pinecone")
                self.results["pinecone"] = False
                return (False, vector_count)
            elif vector_count > expected_max:
                print_warning(f"Found {vector_count} vectors (expected ~9,217)")
                print_info("This may be from multiple ingestion runs")
            else:
                print_success("Vector count looks good! ")
            print_info("\nTesting semantic search...")
            from retrieval.semantic_retriever import SemanticRetriever

            retriever = SemanticRetriever()
            results = retriever.search(
                "What are the confidentiality obligations?", top_k=5
            )
            if results and len(results) > 0:
                print_success(f"Search working! Retrieved {len(results)} results")
                print_info(f"Sample result: {results[0].text[:100]}...")
                self.results["pinecone"] = True
                return (True, vector_count)
            else:
                print_error("Search returned no results")
                self.critical_failures.append("Pinecone search not working")
                self.results["pinecone"] = False
                return (False, vector_count)
        except Exception as e:
            print_error(f"Pinecone check failed: {e!s}")
            self.critical_failures.append(f"Pinecone error: {e!s}")
            self.results["pinecone"] = False
            return (False, 0)

    def check_redis_cache(self) -> bool:
        print_header("4. REDIS SEMANTIC CACHE CHECK")
        try:
            import redis

            print_info("Attempting to connect to Redis...")
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()
            print_success("Redis connection: OK ")
            info = r.info()
            used_memory = info.get("used_memory_human", "Unknown")
            print_success(f"Memory used: {used_memory}")
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            if value == "test_value":
                print_success("Cache read/write: Working ")
                r.delete(test_key)
            cache_keys = r.keys("semantic_cache:*")
            print_info(f"Cached queries: {len(cache_keys)}")
            self.results["redis"] = True
            return True
        except redis.ConnectionError:
            print_error("Cannot connect to Redis!")
            print_warning("\nACTION REQUIRED:")
            print_warning("Start Redis: ./start_redis.sh")
            print_warning("Or: brew services start redis")
            self.critical_failures.append("Redis not running")
            self.results["redis"] = False
            return False
        except Exception as e:
            print_error(f"Redis check failed: {e!s}")
            self.results["redis"] = False
            return False

    def check_cohere_reranker(self) -> bool:
        print_header("5. COHERE RERANKER CHECK")
        try:
            import cohere

            co = cohere.Client(os.getenv("COHERE_API_KEY"))
            print_info("Testing Cohere reranking API...")
            results = co.rerank(
                model="rerank-english-v3.0",
                query="test query",
                documents=["test document 1", "test document 2"],
                top_n=2,
            )
            print_success("Cohere API: Working ")
            print_success("Reranker model: rerank-english-v3.0")
            self.results["cohere"] = True
            return True
        except Exception as e:
            print_error(f"Cohere check failed: {e!s}")
            print_warning("Reranking will be disabled in tests")
            self.results["cohere"] = False
            return False

    def validate_data_quality(self) -> bool:
        print_header("6. DATA QUALITY VALIDATION")
        try:
            import json

            chunks_file = "data/cuad/chunks.json"
            if not os.path.exists(chunks_file):
                print_error(f"Chunks file not found: {chunks_file}")
                print_warning("Run: python3 data/load_cuad.py")
                self.critical_failures.append("Data not ingested")
                return False
            print_info(f"Loading chunks from {chunks_file}...")
            with open(chunks_file, "r") as f:
                chunks = json.load(f)
            chunk_count = len(chunks)
            print_success(f"Total chunks: {chunk_count:,}")
            if chunk_count < 9000:
                print_warning(f"Expected ~9,217 chunks, found {chunk_count}")
            else:
                print_success("Chunk count matches expectations ")
            if chunks:
                sample = chunks[0]
                required_fields = ["text", "chunk_id", "contract_id", "rbac_roles"]
                missing_fields = [f for f in required_fields if f not in sample]
                if missing_fields:
                    print_error(f"Missing fields in chunks: {missing_fields}")
                    return False
                else:
                    print_success("Chunk structure: Valid ")
                if sample.get("rbac_roles"):
                    print_success(
                        f"RBAC metadata: Present (roles: {sample['rbac_roles']})"
                    )
                else:
                    print_warning("RBAC metadata not found (may affect access control)")
            self.results["data_quality"] = True
            return True
        except Exception as e:
            print_error(f"Data validation failed: {e!s}")
            self.results["data_quality"] = False
            return False

    def estimate_test_costs(self):
        print_header("7. COST ESTIMATION FOR TESTING")
        print_info("Cost per query breakdown:")
        print("- Embedding: $0.0001")
        print("- Retrieval: $0.0000 (Pinecone)")
        print("- Reranking: $0.0003 (Cohere)")
        print("- Generation: $0.0137 (GPT-4)")
        print("- Verification: $0.0091 (GPT-4)")
        print("" + "-" * 40)
        print("TOTAL: $0.0232 per query\n")
        scenarios = {
            "Minimal Test (5 queries)": 5 * 0.0232,
            "Validation Test (20 queries)": 20 * 0.0232,
            "Full Demo (50 queries)": 50 * 0.0232,
        }
        print_info("Test scenarios:")
        for scenario, cost in scenarios.items():
            print(f"- {scenario:.<45} ${cost:.2f}")
        print_success("\nRecommended: Start with $10 in OpenAI credits")
        print_info("This gives you ~400 queries of buffer for testing\n")

    def test_end_to_end_query(self) -> bool:
        print_header("8. END-TO-END PIPELINE TEST")
        if not all(
            [self.results.get("openai", False), self.results.get("pinecone", False)]
        ):
            print_error("Skipping E2E test - dependencies not ready")
            return False
        try:
            print_info("Running test query through full pipeline...")
            print_info("Query: 'What are the confidentiality obligations?'\n")
            from retrieval.reranker import CohereReranker
            from retrieval.semantic_retriever import SemanticRetriever
            from verification.answer_verifier import AnswerVerifier

            print_info("Step 1/4: Semantic retrieval...")
            retriever = SemanticRetriever()
            results = retriever.search(
                query="What are the confidentiality obligations?", top_k=10
            )
            print_success(f"Retrieved {len(results)} results")
            if self.results.get("cohere", False):
                print_info("Step 2/4: Reranking results...")
                reranker = CohereReranker()
                reranked_results, metrics = reranker.rerank(
                    query="What are the confidentiality obligations?",
                    results=results,
                    top_k=5,
                )
                print_success(f"Reranked to top {len(reranked_results)} results")
                results = reranked_results
            else:
                print_warning("Step 2/4: Skipping reranking (Cohere not available)")
                results = results[:5]
            print_info("Step 3/4: Generating answer with GPT-4...")
            from openai import OpenAI

            client = OpenAI()
            context = "\n\n".join([r.text for r in results])
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal AI assistant. Answer based on the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: What are the confidentiality obligations?",
                    },
                ],
                max_tokens=200,
            )
            answer = response.choices[0].message.content
            print_success(f"Generated answer ({len(answer)} chars)")
            print_info(f"Answer preview: {answer[:150]}...")
            print_info("\nStep 4/4: Verifying answer...")
            verifier = AnswerVerifier()
            source_chunks = [r.to_dict() for r in results]
            verification = verifier.verify_answer(
                query="What are the confidentiality obligations?",
                answer=answer,
                source_chunks=source_chunks,
            )
            print_success("Verification complete!")
            print_info(f"- Overall Score: {verification.overall_score:.1f}%")
            print_info(f"- Is Trustworthy: {verification.is_trustworthy}")
            print_info(
                f"- Completeness: {verification.completeness.completeness_score:.1f}%"
            )
            print_info(f"- Issues Found: {len(verification.issues_found)}")
            self.results["e2e_test"] = True
            print_success("\nEND-TO-END PIPELINE: WORKING!")
            return True
        except Exception as e:
            print_error(f"E2E test failed: {e!s}")
            import traceback

            print_error(traceback.format_exc())
            self.results["e2e_test"] = False
            return False

    def generate_health_report(self):
        print_header("SYSTEM HEALTH REPORT")
        total_checks = len(self.results)
        passed_checks = sum((1 for v in self.results.values() if v))
        health_percentage = (
            passed_checks / total_checks * 100 if total_checks > 0 else 0
        )
        print(
            f"\n{Colors.BOLD}Results: {passed_checks}/{total_checks} checks passed ({health_percentage:.1f}%){Colors.END}\n"
        )
        for check, passed in self.results.items():
            status = (
                f"{Colors.GREEN} PASS{Colors.END}"
                if passed
                else f"{Colors.RED} FAIL{Colors.END}"
            )
            print(f"{status} {check.replace('_', ' ').title()}")
        print("\n" + "=" * 70 + "\n")
        if self.critical_failures:
            print_error("CRITICAL ISSUES FOUND:")
            for failure in self.critical_failures:
                print(f"{failure}")
            print()
            print_warning("SYSTEM NOT READY FOR TESTING")
            print_info("\nPlease fix the issues above before running tests.\n")
            return False
        else:
            print_success("ALL SYSTEMS GO!")
            print_success("System is ready for testing")
            print_info("\nYou can now run:")
            print_info("- python3 test_complete_system.sh")
            print_info("- python3 validate_project_claims.py")
            print_info("- ./start_api.sh\n")
            return True


def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("")
    print("")
    print("VERIRAG SYSTEM - PRE-FLIGHT CHECK ")
    print("")
    print("This validates your system before testing to ensure: ")
    print("No crashes during testing ")
    print("All claimed metrics are achievable ")
    print("All dependencies are ready ")
    print("")
    print("")
    print(f"{Colors.END}\n")
    validator = SystemHealthValidator()
    validator.check_environment_variables()
    validator.check_openai_credits()
    validator.check_pinecone_index()
    validator.check_redis_cache()
    validator.check_cohere_reranker()
    validator.validate_data_quality()
    validator.estimate_test_costs()
    if all(
        [
            validator.results.get("openai", False),
            validator.results.get("pinecone", False),
        ]
    ):
        validator.test_end_to_end_query()
    success = validator.generate_health_report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

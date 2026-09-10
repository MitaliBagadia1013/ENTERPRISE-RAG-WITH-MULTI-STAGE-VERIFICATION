import argparse
import json
import os
import sys
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from tqdm import tqdm
from retrieval.enhanced_retriever import EnhancedRetriever

load_dotenv()


def load_cuad_qa_sample(sample_size: int = 300) -> list[dict]:
    cuad_path = Path("data/cuad/qa_dataset.json")
    if not cuad_path.exists():
        print("CUAD dataset not found. Run 'python data/load_cuad_qa.py' first!")
        sys.exit(1)
    with open(cuad_path, "r") as f:
        all_qa = json.load(f)
    import random

    random.seed(42)
    sampled = random.sample(all_qa, min(sample_size, len(all_qa)))
    print(f"Loaded {len(sampled)} Q&A pairs from CUAD dataset")
    return sampled


def run_ragas_evaluation(sample_size: int = 300):
    print("=" * 80)
    print("RAGAS EVALUATION - VeriRAG Pipeline")
    print("=" * 80)
    print(f"\nEvaluating {sample_size} queries from CUAD dataset...")
    print(f"Estimated cost: ${sample_size * 0.03:.2f} - ${sample_size * 0.05:.2f}")
    print("\nMetrics to measure:")
    print("1. Context Precision - Retrieval quality")
    print("2. Context Recall - Completeness")
    print("3. Faithfulness - Anti-hallucination")
    print("4. Answer Relevancy - Answer quality")
    print("\n" + "=" * 80)
    print("\nInitializing pipeline components...")
    retriever = EnhancedRetriever(
        use_query_expansion=True, use_hybrid_search=True, use_cohere_reranking=False
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qa_pairs = load_cuad_qa_sample(sample_size)
    evaluation_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    print(f"\nProcessing {len(qa_pairs)} queries...")
    for qa in tqdm(qa_pairs, desc="Evaluating"):
        try:
            retrieval_results = retriever.retrieve(
                query=qa["question"], top_k=3, access_level="user"
            )
            contexts = [chunk.get("text", "") for chunk in retrieval_results]
            context_text = "\n\n".join(contexts)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal contract analyst. Answer based ONLY on the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {qa['question']}\n\nAnswer:",
                    },
                ],
                temperature=0.1,
                max_tokens=300,
            )
            answer = response.choices[0].message.content
            evaluation_data["question"].append(qa["question"])
            evaluation_data["answer"].append(answer)
            evaluation_data["contexts"].append(contexts)
            evaluation_data["ground_truth"].append(qa["answer"])
        except Exception as e:
            print(f"\nError processing query: {e}")
            continue
    dataset = Dataset.from_dict(evaluation_data)
    print(f"\nProcessed {len(evaluation_data['question'])} queries successfully")
    print("\nRunning RAGAS evaluation (this may take 5-10 minutes)...")
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nContext Precision: {results['context_precision']:.2%}")
    print("Are retrieved chunks relevant?")
    print(f"\nContext Recall: {results['context_recall']:.2%}")
    print("Did we get all necessary information?")
    print(f"\nFaithfulness: {results['faithfulness']:.2%}")
    print("Is answer grounded in sources?")
    print(f"\nAnswer Relevancy: {results['answer_relevancy']:.2%}")
    print("Does answer address the question?")
    print("\n" + "=" * 80)
    output_file = f"ragas_evaluation_results_{sample_size}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "sample_size": len(evaluation_data["question"]),
                "metrics": {
                    "context_precision": float(results["context_precision"]),
                    "context_recall": float(results["context_recall"]),
                    "faithfulness": float(results["faithfulness"]),
                    "answer_relevancy": float(results["answer_relevancy"]),
                },
                "evaluation_data": evaluation_data,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_file}")
    print("\nUse these benchmark scores in your resume/portfolio!")
    return results


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation")
    parser.add_argument(
        "--sample-size", type=int, default=300, help="Number of Q&A pairs to evaluate"
    )
    args = parser.parse_args()
    run_ragas_evaluation(sample_size=args.sample_size)


if __name__ == "__main__":
    main()

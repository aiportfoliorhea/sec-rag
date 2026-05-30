import os

from constants import CHUNK_OVERLAP, CHUNK_SIZE
from graph import build_rag_graph
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
import pandas as pd

print(f"\n=== CHUNK CONFIG: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} ===\n")

test_questions = [
    # Answerable
    "Where is JPMorgan Chase headquartered?",
    "What is JPMorgan Chase's ticker symbol and on which exchange does it trade?",
    "What are JPMorgan Chase's three reportable business segments?",
    "How many employees does JPMorgan Chase have globally?",
    "What were JPMorgan Chase's total assets as of December 31, 2025?",
    # Not answerable
    "What is JPMorgan Chase's net interest income for 2025?",
    "What is JPMorgan Chase's CET1 capital ratio?",
    "Who is the CEO of JPMorgan Chase?",
    "What is JPMorgan Chase's total revenue for 2025?",
    "What is JPMorgan Chase's return on equity?",
]

ground_truths = [
    # Answerable
    "JPMorgan Chase is headquartered at 270 Park Avenue, New York, New York 10017.",
    "JPMorgan Chase trades under the ticker symbol JPM on the New York Stock Exchange.",
    "JPMorgan Chase's three reportable business segments are Consumer and Community Banking (CCB), Commercial and Investment Bank (CIB), and Asset and Wealth Management (AWM).",
    "JPMorgan Chase had 318,512 employees globally as of December 31, 2025.",
    "JPMorgan Chase had $4.4 trillion in assets as of December 31, 2025.",
    # Not answerable
    "JPMorgan Chase's net interest income for 2025 is not available in this excerpt.",
    "JPMorgan Chase's CET1 capital ratio is not available in this excerpt.",
    "Jamie Dimon is the CEO of JPMorgan Chase.",
    "JPMorgan Chase's total revenue for 2025 is not available in this excerpt.",
    "JPMorgan Chase's return on equity is not available in this excerpt.",
]

# Generate answers + capture retrieved context
answers = []
contexts = []
rag_graph = build_rag_graph()
for question in test_questions:
    result = rag_graph.invoke({"question": question, "retry_count": 0})
    contexts.append([doc.page_content for doc in result["retrieved_docs"]])
    answer = result["answer"]
    answers.append(answer)

# RAGAS with Claude instead of OpenAI (already paid for the key :/)
judge_llm = LangchainLLMWrapper(
    ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
    )
)

judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Apply to each metric
for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
    metric.llm = judge_llm

answer_relevancy.embeddings = judge_embeddings

# Build dataset
data = {
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(results)

df = results.to_pandas()

# Split into answerable vs not
df["answerable"] = (
    ["Yes"] * 5 + ["No"] * 5
)

print("=== FULL RESULTS ===")
print(df[["user_input", "faithfulness", "answer_relevancy",
          "context_precision", "context_recall", "answerable"]].to_string())

print("\n=== AVERAGES BY ANSWERABILITY ===")
print(df.groupby("answerable")[
    ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
].mean().round(3))

df["chunk_size"] = CHUNK_SIZE
df["chunk_overlap"] = CHUNK_OVERLAP
df.to_csv(f"eval_results_chunk{CHUNK_SIZE}.csv", index=False)
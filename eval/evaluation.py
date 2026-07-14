import json
import math
import asyncio
import math
import asyncio

import sys, types
# ragas <=0.4.3 hard-imports langchain_community.chat_models.vertexai,
# removed in the langchain 1.x community restructure (ragas #2745).
# Anthropic-only -> ChatVertexAI never instantiated. Stub it.
_m = "langchain_community.chat_models.vertexai"
if _m not in sys.modules:
    _stub = types.ModuleType(_m)
    class ChatVertexAI: pass
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules[_m] = _stub

from src.graph import build_rag_graph
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent 

FAITHFULNESS_GATE = 0.85  

def load_test_set(path="test_set.json"):
    with open(REPO_ROOT / "test" / path) as f:
        return json.load(f)
    
async def score_faithfulness(scorer, question, answer, contexts):
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    return await scorer.single_turn_ascore(sample)

async def run_eval(test_set):
    graph = build_rag_graph()
    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024))
    scorer = Faithfulness(llm=evaluator_llm)
    results = []
    for case in test_set:
        state = graph.invoke({
            "question": case["question"],
            "retry_count": 0,           
        })
        contexts = [d.page_content for d in state["retrieved_docs"]]
        score = await score_faithfulness(
            scorer, case["question"], state["answer"], contexts
        )
        results.append({
            "question": case["question"],
            "answerable": case["answerable"],
            "answer": state["answer"],
            "faithfulness": score,   # may be NaN on unanswerable
        })
    return results


def gate(results):
    # answerable-only; drop NaN explicitly and surface it
    answerable = [r for r in results if r["answerable"]]
    scored = [r for r in answerable if not math.isnan(r["faithfulness"])]
    dropped = len(answerable) - len(scored)
    if not scored:
        raise SystemExit("FAIL: no scorable answerable questions")
    mean_faith = sum(r["faithfulness"] for r in scored) / len(scored)
    print(f"Answerable faithfulness: {mean_faith:.3f} "
          f"(n={len(scored)}, {dropped} NaN dropped)")
    if mean_faith < FAITHFULNESS_GATE:
        raise SystemExit(f"FAIL: {mean_faith:.3f} < {FAITHFULNESS_GATE}")
    print("PASS")

if __name__ == "__main__":
    ts = load_test_set()
    out = asyncio.run(run_eval(ts))
    print(f"Ran {len(out)} cases")
    print("\n--- per question ---")
    for r in out:
        tag = "ANS  " if r["answerable"] else "unans"
        preview = r["answer"].splitlines()[0][:70] if r["answer"] else ""
        print(f"[{tag}] {r['faithfulness']:.3f}  {r['question'][:55]}  | {preview}")
    gate(out)
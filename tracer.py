"""
Purpose: LangSmith is the primary observability layer, but it's an external
dependency. Tracer will give you a local, queryable, structured log of every query:
per-node timing, LLM token usage + cost, and Cohere rerank scores — written
to a JSONL file.
"""

import json
import time
import os
from contextlib import contextmanager
from datetime import datetime, timezone

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_results")
os.makedirs(TRACE_DIR, exist_ok=True)
TRACE_FILE = os.getenv("TRACE_LOG_PATH", os.path.join(TRACE_DIR, "traces.jsonl"))

# Verified 2026-07-07 against platform.claude.com/docs/en/about-claude/pricing
# and cross-referenced against 3 independent pricing trackers. $ per million tokens.
PRICING_PER_1M_TOKENS = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_1M_TOKENS.get(model)
    if not rates:
        return 0.0
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


@contextmanager
def timed(trace: dict, node_name: str):
    """Wrap a node body to record its wall-clock duration into trace['timings']."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        trace.setdefault("timings", []).append({
            "node": node_name,
            "duration_ms": round(duration_ms, 2),
        })


def log_llm_usage(trace: dict, node_name: str, model: str, response) -> None:
    """Pull token usage off a ChatAnthropic response and record cost.

    ChatAnthropic (langchain-anthropic) attaches `usage_metadata` to AIMessage
    responses: {'input_tokens': int, 'output_tokens': int, 'total_tokens': int}.
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = estimate_cost(model, input_tokens, output_tokens)
    trace.setdefault("token_usage", []).append({
        "node": node_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    })


def log_rerank_scores(trace: dict, cohere_response) -> None:
    """Record Cohere rerank relevance scores for this retrieval call."""
    trace.setdefault("rerank_scores", []).append([
        {"index": r.index, "relevance_score": r.relevance_score}
        for r in cohere_response.results
    ])


def write_trace(trace: dict, question: str, final_state: dict) -> dict:
    """Flush the accumulated trace dict for one full query (incl. retries) to disk."""
    total_cost = sum(t["cost_usd"] for t in trace.get("token_usage", []))
    total_duration = sum(t["duration_ms"] for t in trace.get("timings", []))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retry_count": final_state.get("retry_count"),
        "validation_score": final_state.get("validation_score"),
        "total_duration_ms": round(total_duration, 2),
        "total_cost_usd": round(total_cost, 6),
        "timings": trace.get("timings", []),
        "token_usage": trace.get("token_usage", []),
        "rerank_scores": trace.get("rerank_scores", []),
    }
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
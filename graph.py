from langgraph.graph import StateGraph, END
from typing import TypedDict
from retriever import load_vector_store
from client import cohere_client
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from tracer import timed, log_llm_usage, log_rerank_scores


llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512)
llm_short = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=256)

# 1. Define state
class SecRagState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_docs: list
    answer: str
    validation_score: float
    retry_count: int
    trace: dict  # accumulates timings/token_usage/rerank_scores across the whole run, incl. retries

# 2. Query rewriter node
def rewrite_query(state: SecRagState) -> SecRagState:
    trace = state.setdefault("trace", {})
    with timed(trace, "rewrite_query"):
        response = llm_short.invoke([
            HumanMessage(content=f"""Rewrite this question to be a better search query for retrieving relevant chunks from a JPMorgan 10-K SEC filing. 
                                 Return only the rewritten query, nothing else. Question: {state['question']}""")])
    state["rewritten_query"] = response.content.strip()
    log_llm_usage(trace, "rewrite_query", "claude-sonnet-4-6", response)
    return state

# 3. Retrieval node
def retrieve(state: SecRagState) -> SecRagState:
    trace = state["trace"] 
    with timed(trace, "retrieve"):
        retriever = load_vector_store()
        docs = retriever.invoke(state["rewritten_query"])
        cohere_response = cohere_client.rerank(top_n=3, documents=[doc.page_content for doc in docs], query=state["question"])
        top_docs = [docs[result.index] for result in cohere_response.results]
    state["retrieved_docs"] = top_docs
    log_rerank_scores(trace, cohere_response)
    return state

# 4. Answer node
def generate_answer(state: SecRagState) -> SecRagState:
    trace = state["trace"] 
    context = "\n\n".join([d.page_content for d in state["retrieved_docs"]])
    with timed(trace, "generate_answer"):
        response = llm.invoke([
        SystemMessage(content="Act as a SEC filing assistant. Answer the question using only the context provided. If the answer is not in the context, say 'I don't have that information in the document'"),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer:")
        ])
    state["answer"] = response.content.strip()
    log_llm_usage(trace, "generate_answer", "claude-sonnet-4-6", response)
    return state

# 5. validate faithfulness
def validate_answer(state: SecRagState) -> SecRagState:
    trace = state["trace"] 
    state["retry_count"] += 1
    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])  
    with timed(trace, "validate_answer"):
        response = llm.invoke([
            SystemMessage(content="Act as a SEC filing assistant. Check if the given answer is present or is derived from the given Context. Your job is evaluate the faithfulness of this answer through validation score ranging from 0-1. If the context does not contain sufficient evidence for the answer indicate that with a low validation score. Return the score only, do not add any text to it."),
            HumanMessage(content=f"Answer:\n{state['answer']}\n\nContext: {context}")
        ])
    try:
        state["validation_score"] = float(response.content.strip())
    except:
        state["validation_score"] = 0.6
    log_llm_usage(trace, "validate_answer", "claude-sonnet-4-6", response)
    return state

# 6. retrigger retrival of docs conditionally
def retrigger_retrieval(state: SecRagState) -> SecRagState:
    if state["validation_score"] < 0.7 and state["retry_count"] <= 3:
        return "retrieve"
    return "end"

# 7. Build graph
def build_rag_graph():
    graph = StateGraph(SecRagState)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_answer)
    graph.add_node("validate", validate_answer)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", retrigger_retrieval, {"retrieve": "retrieve", "end": END})
    
    return graph.compile()
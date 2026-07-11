from graph import build_rag_graph

def test_pipeline():
    graph = build_rag_graph()
    result = graph.invoke({"question": "Where is JPMorgan Chase headquartered?", "retry_count": 0})
    
    assert result["answer"] is not None
    assert len(result["retrieved_docs"]) > 0
    assert "New York" in result["answer"]
    
    print("Pipeline working")
    print(f"Rewritten query: {result['rewritten_query']}")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    test_pipeline()
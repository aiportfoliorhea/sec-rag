from mcp.server.fastmcp import FastMCP
from graph import build_rag_graph

sec_rag = FastMCP("sec_rag")

@sec_rag.tool()
def query_sec_rag(query: str) -> str:
    rag_graph = build_rag_graph() 
    result = rag_graph.invoke({"question": query, "retry_count": 0})
    return result["answer"]

if __name__ == "__main__":
    sec_rag.run()
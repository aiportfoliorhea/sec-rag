from anthropic import Anthropic
import streamlit as st
from graph import build_rag_graph
from retriever import load_vector_store

client = Anthropic()
rag_graph = build_rag_graph() 

def ask_loanbot(question):
    result = rag_graph.invoke({"question": question, "retry_count": 0})
    return result["answer"], result["retrieved_docs"], result["rewritten_query"], result["validation_score"]


st.title("SEC Document Assistant")
if "input" not in st.session_state:
    st.session_state.input = ""
with st.expander("Sample questions"):
    if st.button("What were JPMorgan Chase's total assets as of December 31, 2025?"):
        st.session_state.input = "What were JPMorgan Chase's total assets as of December 31, 2025?"
    if st.button( "How many employees does JPMorgan Chase have globally?"):
        st.session_state.input =    "How many employees does JPMorgan Chase have globally?"
    if st.button( "Where is JPMorgan Chase headquartered?"):
        st.session_state.input =  "Where is JPMorgan Chase headquartered?"
      
input = st.text_input("Type your question", key="input")
  
if st.button("Ask"):
    response, retrieved_docs, rewritten_query, validation_score = ask_loanbot(input)
    st.write(response)
    st.write("Agent validation score is:", validation_score)
    with st.expander("Retrieved Docs"):
        st.write(retrieved_docs)
    st.write("Rewritten query:", rewritten_query)

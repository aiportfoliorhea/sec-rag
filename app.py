
import streamlit as st
from graph import build_rag_graph

rag_graph = build_rag_graph() 

def ask_sec_rag(question):
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
    if st.session_state.input.strip() == "":
        st.warning("Please enter a question.")
        st.stop()
    try:
        with st.spinner("Thinking..."):
            response, retrieved_docs, rewritten_query, validation_score = ask_sec_rag(input)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
    st.write(response)
    with st.expander("Retrieved Docs"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content)
        st.write("Agent validation score is:", validation_score)
    st.write("Rewritten query:", rewritten_query)

import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import the parts from your logic file
from chatbot import ingestion_chain, prompt, p_llm, format_docs

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("📺 YouTube Video Chatbot")

# --- 1. INITIALIZE SESSION STATE ---
# This is the "brain" of the app that survives reruns
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- 2. SIDEBAR: INGESTION ---
with st.sidebar:
    st.header("Step 1: Ingest Video")
    youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/...")
    
    if st.button("Process & Index Video", use_container_width=True):
        if youtube_url:
            with st.spinner("Processing transcript..."):
                try:
                    # Execute Ingestion Chain
                    results = ingestion_chain.invoke({"url": youtube_url})
                    
                    if results.get("retriever"):
                        st.session_state.retriever = results["retriever"]
                        st.success("Video indexed!")
                    else:
                        st.error(f"Error: {results.get('error')}")
                except Exception as e:
                    st.error(f"Failed to process: {e}")
        else:
            st.warning("Please enter a URL.")

# --- 3. MAIN AREA: CHAT ---
st.header("Step 2: Chat with Video")

# Check if we have a retriever in memory
if st.session_state.retriever is None:
    st.info("👈 Please process a YouTube video in the sidebar to start chatting.")
else:
    st.write("---")
    user_question = st.text_input("Ask a question about the video content:")

    if st.button("Get Answer", type="primary"):
        if user_question:
            with st.spinner("Searching transcript and generating answer..."):
                try:
                    # BUILD THE QA CHAIN
                    # We use the retriever currently stored in session_state
                    qa_chain = (
                        {
                            "context": st.session_state.retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough()
                        }
                        | prompt
                        | p_llm
                        | StrOutputParser()
                    )
                    
                    # EXECUTE
                    answer = qa_chain.invoke(user_question)
                    
                    # DISPLAY
                    st.markdown("### Answer")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"An error occurred during Q&A: {e}")
        else:
            st.warning("Please type a question first.")
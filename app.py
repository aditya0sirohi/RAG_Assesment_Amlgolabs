"""
RAG Chatbot - Streamlit UI
"""
import time
import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Custom CSS for better UX
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>");
        background-size: contain;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
st.title("🤖 RAG Chatbot")
st.markdown("*Ask questions about your documents. Powered by semantic search and AI.*")

if "pipeline" not in st.session_state:
    with st.spinner("Initializing pipeline..."):
        st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    
    import config
    st.markdown(f"**Model:** `{config.MODEL_NAME.split('/')[-1]}`")
    st.markdown(f"**Temp:** {config.TEMPERATURE}")
    st.markdown(f"**Max Tokens:** {config.MAX_TOKENS}")
    st.markdown(f"**Top-K Chunks:** {config.TOP_K}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("🔍 Searching documents & generating response..."):
                stream, chunks = st.session_state.pipeline.stream_run(prompt)

            # Word-level streaming for guaranteed visible effect
            for token in stream:
                words = token.split(" ")
                for word in words:
                    if word:  # Skip empty strings
                        full_response += word + " "
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.04)  # Word-by-word for smooth UX regardless of token batching

            placeholder.markdown(full_response)

            with st.expander("📄 View Source Chunks", expanded=False):
                if chunks:
                    for i, chunk in enumerate(chunks, 1):
                        st.divider()
                        # Handle both Document objects and strings
                        if hasattr(chunk, 'page_content'):
                            content = chunk.page_content
                            page = chunk.metadata.get('page', 'N/A') if hasattr(chunk, 'metadata') else 'N/A'
                        else:
                            content = chunk
                            page = 'N/A'
                        
                        st.markdown(f"**📌 Chunk {i}** — Page {page}")
                        st.info(content[:600] + "..." if len(content) > 600 else content)
                else:
                    st.warning("No source chunks retrieved for this query.")

        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            placeholder.error(error_msg)
            full_response = error_msg
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
            st.stop()

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )


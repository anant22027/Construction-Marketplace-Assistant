"""
Streamlit Chatbot Interface for Construction Marketplace RAG Assistant
"""

import streamlit as st
import os
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Construction Marketplace Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'index_built' not in st.session_state:
    st.session_state.index_built = False

# Title and description
st.title("🏗️ Construction Marketplace Assistant")
st.markdown("""
This AI assistant answers questions about construction projects using internal documents including policies, FAQs, and specifications.
The assistant uses Retrieval-Augmented Generation (RAG) to provide grounded, accurate answers.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Get your free API key from https://openrouter.ai"
    )
    
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Model selection
    model_options = {
        "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B (Free) - Recommended",
        "mistralai/mistral-7b-instruct:free": "Mistral 7B (Free)",
        "google/gemini-flash-1.5:free": "Gemini Flash 1.5 (Free)",
        "google/gemini-flash-1.5-8b:free": "Gemini Flash 1.5 8B (Free) - May not work"
    }
    
    selected_model = st.selectbox(
        "LLM Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0  # Default to Llama which is more reliable
    )
    
    # Top-k retrieval
    top_k = st.slider("Number of chunks to retrieve", 1, 5, 3)
    
    st.divider()
    
    # Initialize/rebuild index button
    if st.button("🔄 Initialize/Reload RAG Pipeline", type="primary"):
        if not api_key:
            st.error("Please enter your OpenRouter API key first")
        else:
            with st.spinner("Loading documents and building index..."):
                try:
                    rag = RAGPipeline(openrouter_api_key=api_key, openrouter_model=selected_model)
                    
                    # Try to load existing index
                    if os.path.exists("construction_index.index"):
                        try:
                            rag.load_index("construction_index")
                            st.success("Loaded existing index")
                        except:
                            # If loading fails, rebuild
                            documents = rag.load_documents("documents")
                            chunks = rag.chunk_documents(documents)
                            rag.build_index(chunks)
                            rag.save_index("construction_index")
                            st.success("Built new index")
                    else:
                        # Build new index
                        documents = rag.load_documents("documents")
                        chunks = rag.chunk_documents(documents)
                        rag.build_index(chunks)
                        rag.save_index("construction_index")
                        st.success("Built new index")
                    
                    st.session_state.rag_pipeline = rag
                    st.session_state.index_built = True
                except Exception as e:
                    st.error(f"Error initializing pipeline: {str(e)}")
    
    if st.session_state.index_built:
        st.success("✅ RAG Pipeline Ready")
    else:
        st.warning("⚠️ Please initialize the RAG pipeline")

# Main chat interface
if st.session_state.index_built and st.session_state.rag_pipeline:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show retrieved chunks for user queries
            if message["role"] == "user" and "retrieved_chunks" in message:
                with st.expander("📄 Retrieved Context", expanded=False):
                    for i, (chunk, metadata, distance) in enumerate(message["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i}** (Source: `{metadata['source']}`, Distance: {distance:.4f})")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about construction projects..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving information and generating answer..."):
                try:
                    result = st.session_state.rag_pipeline.query(prompt, top_k=top_k)
                    
                    # Store retrieved chunks with user message
                    st.session_state.messages[-1]["retrieved_chunks"] = result['retrieved_chunks']
                    
                    # Display retrieved chunks
                    with st.expander("📄 Retrieved Context", expanded=True):
                        st.markdown("**The following document chunks were used to generate the answer:**")
                        for i, (chunk, metadata, distance) in enumerate(result['retrieved_chunks'], 1):
                            st.markdown(f"**Chunk {i}** (Source: `{metadata['source']}`, Similarity Score: {1/(1+distance):.4f})")
                            st.text(chunk)
                            st.divider()
                    
                    # Display answer
                    st.markdown("**Answer:**")
                    st.markdown(result['answer'])
                    
                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['answer']
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

else:
    # Welcome message when pipeline not initialized
    st.info("👈 Please configure and initialize the RAG pipeline using the sidebar.")
    
    st.markdown("""
    ### How to use:
    1. Enter your OpenRouter API key in the sidebar (get one free at https://openrouter.ai)
    2. Select an LLM model
    3. Click "Initialize/Reload RAG Pipeline" to load documents and build the index
    4. Start asking questions about construction projects!
    
    ### Example Questions:
    - What factors affect construction project delays?
    - What safety protocols must be followed on construction sites?
    - How long does a typical residential construction project take?
    - What permits are required for construction projects?
    - What are the concrete specifications for foundations?
    """)


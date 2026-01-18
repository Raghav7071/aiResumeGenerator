import streamlit as st
import os
import sys
import logging
import json
from dotenv import load_dotenv

# Add the current directory to sys.path so we can import from the backend folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.chroma_store import ChromaVectorStore
from backend.pipeline.ingestion import extract_text_from_pdf, chunk_text
from backend.pipeline.retrieval import RetrievalPipeline
from backend.embeddings import embedding_model

# Page Configuration
st.set_page_config(
    page_title="DocuMind | AI Knowledge Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Environment Variables
load_dotenv()

# --- CUSTOM CSS FOR UI PARITY ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f0fdf4 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f0c 0%, #022c22 100%) !important;
        border-right: none !important;
        width: 300px !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* Sidebar 'New Chat' Button Style */
    div[data-testid="stSidebar"] button {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        padding: 10px 15px !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stSidebar"] button:hover {
        background-color: #022c22 !important;
        color: white !important;
    }

    /* Special Green Button for New Chat */
    div[data-testid="stSidebar"] div.stButton:first-child button {
        background-color: #14532d !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    div[data-testid="stSidebar"] div.stButton:first-child button:hover {
        background-color: #166534 !important;
        transform: translateY(-1px) !important;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0 !important;
        margin-bottom: 24px !important;
    }

    /* Assistant Message (White) */
    div[data-testid="stChatMessageAssistant"] {
        display: flex !important;
        justify-content: flex-start !important;
    }
    
    div[data-testid="stChatMessageAssistant"] > div {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #f1f5f9 !important;
        border-radius: 20px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        max-width: 85% !important;
    }

    /* User Message (Dark Green) */
    div[data-testid="stChatMessageUser"] {
        display: flex !important;
        justify-content: flex-end !important;
    }

    div[data-testid="stChatMessageUser"] > div {
        background-color: #14532d !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important;
        max-width: 85% !important;
    }

    /* Chat Input Bar (Floating Look) */
    div[data-testid="stChatInput"] {
        background-color: white !important;
        border: 2px solid #f1f5f9 !important;
        border-radius: 24px !important;
        padding: 4px 8px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        bottom: 30px !important;
    }

    /* Header Styling */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px) !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    
    .stHeaderTitle {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* Hide standard streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}

</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "view" not in st.session_state:
    st.session_state.view = "chat"

@st.cache_resource
def get_vector_store():
    persist_dir = os.path.join(os.path.dirname(__file__), "backend", "chroma_db")
    return ChromaVectorStore(persist_directory=persist_dir)

@st.cache_resource
def get_retrieval_pipeline(_vector_store):
    api_key = os.getenv("GROQ_API_KEY")
    return RetrievalPipeline(api_key=api_key, vector_store=_vector_store) if api_key else None

vector_store = get_vector_store()
retrieval_pipeline = get_retrieval_pipeline(vector_store)

# --- SIDEBAR (Logic & Nav) ---
with st.sidebar:
    # Logo & Title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("🟢") # Placeholder for icon
    with col2:
        st.markdown("### DocuMind AI")
    
    st.divider()

    # Navigation Buttons (Mimicking React)
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.view = "chat"
        st.rerun()

    if st.button("📁 Knowledge Base"):
        st.session_state.view = "knowledge"
        st.rerun()

    if st.button("🔖 Saved Citations"):
        st.session_state.view = "citations"
        st.rerun()

    if st.button("📊 Analytics"):
        st.session_state.view = "analytics"
        st.rerun()

    if st.button("📥 Export Data"):
        st.session_state.view = "export"
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='font-size: 10px; font-weight: bold; color: #64748b; letter-spacing: 1px;'>RECENTS</p>", unsafe_allow_html=True)
    
    docs = vector_store.list_documents()
    if docs:
        for doc in docs[:5]:
            st.markdown(f"<p style='font-size: 12px; color: #94a3b8; margin-bottom: 5px;'>📄 {doc['filename'][:20]}...</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size: 12px; color: #475569; font-style: italic;'>No recent uploads</p>", unsafe_allow_html=True)

    # Status at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.container():
        st.markdown("**System Status**")
        if os.getenv("GROQ_API_KEY"):
            st.success("API Ready", icon="✅")
        else:
            st.warning("Key Missing", icon="⚠️")

# --- MAIN CONTENT ---
if st.session_state.view == "chat":
    # Custom Header
    st.markdown("## DocuMind Knowledge Engine")
    st.divider()

    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ **GROQ_API_KEY is missing.** Please add it to your Streamlit Secrets (Settings -> Secrets) to enable AI chat.")

    # Display Messages (Light Theme Styles)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("Sources"):
                    for cite in message["citations"]:
                        st.write(f"- {cite.get('filename')} (Page {cite.get('page_number')})")

    # Bottom Chat Input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Rerun to show user message immediately

    # Handle logic if last message is user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            if not retrieval_pipeline:
                st.error("AI engine not ready.")
            else:
                with st.spinner("Researching..."):
                    try:
                        res = retrieval_pipeline.run(user_prompt)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": res["answer"], 
                            "citations": res["citations"]
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

elif st.session_state.view == "knowledge":
    st.markdown("## Knowledge Base")
    st.divider()
    
    # Upload Sector
    uploaded_file = st.file_uploader("Upload PDF Documents", type=["pdf"])
    if uploaded_file:
        if st.button("🚀 Process & Index"):
            with st.status(f"Indexing {uploaded_file.name}...", expanded=True) as status:
                file_content = uploaded_file.read()
                pages = extract_text_from_pdf(file_content, uploaded_file.name)
                chunks = chunk_text(pages)
                embeddings = embedding_model.generate_batch([c["text"] for c in chunks])
                if vector_store.insert(embeddings, chunks):
                    status.update(label="Index Complete!", state="complete", expanded=False)
                    st.balloons()
                    st.rerun()

    # List & Management
    st.write("---")
    docs = vector_store.list_documents()
    if docs:
        for doc in docs:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"📄 **{doc['filename']}** ({doc['page_count']} pages)")
            if c2.button("🗑️", key=f"del_{doc['filename']}"):
                vector_store.delete_document(doc['filename'])
                st.rerun()
    else:
        st.info("Knowledge base is empty.")

else:
    # Placeholder for other views
    st.title(st.session_state.view.title())
    st.markdown("### Feature Coming Soon")
    st.info(f"We are working hard to bring the {st.session_state.view} module to your deployment.")
    if st.button("⬅️ Back to Chat"):
        st.session_state.view = "chat"
        st.rerun()


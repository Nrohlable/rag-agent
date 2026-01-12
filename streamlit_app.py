import streamlit as st
import os
from modules.rag_agent import RAGAgent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

def main():
    st.title("📚 RAG Document Chat")
    st.sidebar.header("Document Source")
    
    # Reset button
    if st.sidebar.button("Reset App"):
        st.session_state.rag_agent = None
        st.session_state.document_uploaded = False
        st.session_state.conversation_history = []
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    # Check if OpenAI API key is set
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("❌ OPENAI_API_KEY environment variable is not set!")
        st.info("Please set your OpenAI API key in the .env file.")
        return
    elif len(openai_key) < 20:
        st.error("❌ Invalid OpenAI API key detected!")
        st.info("Please update your OPENAI_API_KEY in the .env file with a valid key.")
        return
    else:
        # Show configuration info
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        max_tokens = os.getenv("OPENAI_MAX_TOKENS", "2048")
        st.sidebar.success("✅ Configuration loaded successfully")
        st.sidebar.info(f"Model: {model} | Max tokens: {max_tokens}")
    
    # Document source selection
    doc_source = st.sidebar.radio(
        "Choose document source:",
        ["Upload your own document", "Use internal knowledge base"],
        help="Select whether to upload your own file or use the internal documents"
    )
    
    if doc_source == "Use internal knowledge base":
        handle_internal_documents()
    else:
        handle_file_upload()

    # Show chat interface
    show_chat_interface()
def handle_file_upload():
    """Handle file upload functionality"""
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=['txt', 'pdf'],
        help="Upload a PDF or TXT document to chat with"
    )
    
    if uploaded_file is not None and not st.session_state.document_uploaded:
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.write(f"File saved temporarily as: {temp_path}")
                st.write(f"File size: {len(uploaded_file.getvalue())} bytes")
                
                # Initialize RAG agent and ingest document
                if st.session_state.rag_agent is None:
                    st.write("Initializing RAG agent...")
                    st.session_state.rag_agent = RAGAgent(default_pdf_path=None)
                
                st.write("Processing document...")
                st.session_state.rag_agent.ingest_document(temp_path)
                st.session_state.document_uploaded = True
                
                # Clean up
                os.remove(temp_path)
                st.sidebar.success(f"Document '{uploaded_file.name}' uploaded successfully!")
                st.rerun()  # Refresh the app to show the chat interface
                
            except Exception as e:
                import traceback
                error_str = str(e)
                error_details = traceback.format_exc()
                
                # Check for specific error types
                if "401" in error_str or "invalid_api_key" in error_str or "Incorrect API key" in error_str:
                    st.sidebar.error("❌ Invalid OpenAI API Key!")
                    st.error("**Authentication Error**: Your OpenAI API key is invalid or expired.")
                    st.info("Please update your OPENAI_API_KEY in the .env file with a valid key from https://platform.openai.com/account/api-keys")
                elif "appears to contain no readable text" in error_str or "appears to be empty" in error_str:
                    st.sidebar.error("❌ PDF Processing Error!")
                    st.error("**PDF Issue**: The PDF appears to contain no readable text.")
                    st.info("**Possible solutions:**\n"
                           "• Try converting the PDF to a text file\n"
                           "• Use a PDF with actual text (not scanned images)\n" 
                           "• If it's a scanned document, use OCR to extract text first\n"
                           "• Try uploading a different document")
                elif "list index out of range" in error_str:
                    st.sidebar.error("❌ Document Processing Error!")
                    st.error("**Document Error**: The uploaded document could not be processed properly.")
                    st.info("This might be due to an empty document or unsupported format. Please try uploading a different document.")
                else:
                    st.sidebar.error(f"Error processing document: {error_str}")
                    st.error("Detailed error information:")
                    st.code(error_details)
                
                # Clean up temp file if it exists
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)

def handle_internal_documents():
    """Handle internal documents functionality"""
    if st.session_state.rag_agent is None:
        st.session_state.rag_agent = RAGAgent(default_pdf_path=None)
    
    # Get available internal documents
    available_docs = st.session_state.rag_agent.get_available_internal_documents()
    
    if available_docs:
        st.sidebar.success(f"📁 Found {len(available_docs)} internal documents:")
        for doc in available_docs:
            st.sidebar.text(f"• {doc}")
        
        if st.sidebar.button("Load Internal Knowledge Base") and not st.session_state.document_uploaded:
            with st.spinner("Loading internal documents..."):
                try:
                    loaded_files = st.session_state.rag_agent.load_internal_documents()
                    if loaded_files:
                        st.session_state.document_uploaded = True
                        st.sidebar.success(f"✅ Loaded {len(loaded_files)} documents successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ No documents could be loaded")
                except Exception as e:
                    st.sidebar.error(f"Error loading documents: {str(e)}")
    else:
        st.sidebar.warning("📁 No internal documents found")
        st.sidebar.info("Internal documents should be placed in the 'data' folder as PDF or TXT files")

def show_chat_interface():
    if st.session_state.document_uploaded and st.session_state.rag_agent:
        st.header("Chat with your document")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for i, (question, answer) in enumerate(st.session_state.conversation_history):
                with st.container():
                    st.markdown(f"**🙋 Question {i+1}:** {question}")
                    st.success("**Answer:**")
                    st.markdown(f"{answer}")
                    st.markdown("---")
        
        # Chat input with form to prevent auto-rerun
        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_area(
                "Ask a question about your document:", 
                placeholder="Type your question here...\n\nYou can ask follow-up questions, request more details, or ask about specific parts of the document.",
                height=100,
                help="The text area will expand automatically as you type more text."
            )
            submitted = st.form_submit_button("Send")
            
            if submitted and user_question.strip():
                try:
                    with st.spinner("Generating response..."):
                        # Use the persistent thread_id for conversation continuity
                        response = st.session_state.rag_agent.run(user_question.strip(), st.session_state.thread_id)
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append((user_question.strip(), response))
                        
                        # Rerun to show updated history and clear form
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Show latest response if available
        if st.session_state.conversation_history:
            latest_q, latest_a = st.session_state.conversation_history[-1]
        
        # Clear conversation button
        if st.session_state.conversation_history:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.conversation_history = []
                import uuid
                st.session_state.thread_id = str(uuid.uuid4())  # Start fresh thread
                st.rerun()
    
    else:
        st.info("👆 Please choose a document source from the sidebar to get started!")
        st.markdown("### Available Options:")
        st.markdown("- **Upload your own document**: Upload a PDF or TXT file to chat with")
        st.markdown("- **Use internal knowledge base**: Access our collection of company documents")

if __name__ == "__main__":
    main()
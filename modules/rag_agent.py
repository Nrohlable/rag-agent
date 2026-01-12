

import os
import glob
import sys
from typing import List, Sequence, Annotated, TypedDict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import uuid

# Import prompts config
from .config.prompts import config

# Load environment variables
load_dotenv()

class RAGAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RAGAgent:
    def __init__(self, default_pdf_path: Optional[str] = None):
        # Load default PDF path from environment if not provided
        if default_pdf_path is None:
            default_pdf_path = os.getenv("DEFAULT_PDF_PATH")
            
        # Initialize embeddings with environment settings
        embedding_model = os.getenv("EMBEDDING_MODEL")
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize text splitter with environment settings
        chunk_size = int(os.getenv("CHUNK_SIZE"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.vectorstore = None
        self.memory = MemorySaver()
        self.debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # Initialize LLM with environment settings
        model = os.getenv("OPENAI_MODEL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS"))
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.default_pdf_path = default_pdf_path
        
        # Data folder path for internal documents
        self.data_folder = os.getenv("DATA_FOLDER", "data")
        
        self._init_vectorstore_with_default()
        self.tools = []
        if self.vectorstore:
            self.tools = [self.create_retriever_tool()]
            self.llm = self.llm.bind_tools(self.tools)
        self.app = self._build_graph()

    def _init_vectorstore_with_default(self):
        if self.default_pdf_path and os.path.exists(self.default_pdf_path):
            loader = PyPDFLoader(self.default_pdf_path)
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vectorstore = None

    def clean_pdf_text(self, text: str) -> str:
        """
        Clean and normalize PDF text that may have formatting issues
        """
        import re
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newlines
        
        # Fix common PDF extraction issues where words are split by single spaces/newlines
        # Pattern: single letter/word followed by newline and space
        text = re.sub(r'(\w)\s*\n\s*(?=\w)', r'\1 ', text)
        
        # Fix bullet points and special characters
        text = re.sub(r'●\s*', '• ', text)
        text = re.sub(r'•\s*', '• ', text)
        
        # Join lines that seem to be part of the same sentence/phrase
        # Look for lines that end with lowercase and start with lowercase (likely continuation)
        lines = text.split('\n')
        cleaned_lines = []
        current_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_line:
                    cleaned_lines.append(current_line)
                    current_line = ""
                cleaned_lines.append("")
                continue
                
            # If current line ends with lowercase/punctuation and next line starts with lowercase,
            # they're likely the same sentence
            if (current_line and len(current_line) > 0 and 
                (current_line[-1].islower() or current_line[-1] in ',-') and
                line and len(line) > 0 and line[0].islower()):
                current_line += " " + line
            else:
                if current_line:
                    cleaned_lines.append(current_line)
                current_line = line
        
        if current_line:
            cleaned_lines.append(current_line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
        cleaned_text = re.sub(r'\n\s+', '\n', cleaned_text)  # Leading spaces after newlines
        
        return cleaned_text.strip()

    def ingest_document(self, file_path: str):
        # Use for user-uploaded TXT or PDF
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            print(f"Processing file: {file_path} with extension: {ext}")
            
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                print(f"Loaded {len(docs)} pages from PDF")
                
                # Check each page for content and filter out empty pages
                non_empty_docs = []
                for i, doc in enumerate(docs):
                    content = doc.page_content.strip()
                    if content and len(content) > 10:  # Must have at least 10 characters
                        # Clean the PDF text to fix formatting issues
                        cleaned_content = self.clean_pdf_text(content)
                        doc.page_content = cleaned_content
                        non_empty_docs.append(doc)
                        print(f"Page {i+1}: {len(content)} characters (cleaned to {len(cleaned_content)} chars)")
                    else:
                        print(f"Page {i+1}: Empty or too short ({len(content)} characters) - skipping")
                
                docs = non_empty_docs
                
                if not docs:
                    raise Exception("PDF appears to contain no readable text. This could be due to: "
                                  "1) Scanned images without OCR text, "
                                  "2) Protected/encrypted content, "
                                  "3) Complex formatting. "
                                  "Try uploading a text-based PDF or convert it to TXT format.")
                    
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                from langchain_core.documents import Document
                docs = [Document(page_content=text)]
                print(f"Loaded text document with {len(text)} characters")
            else:
                raise Exception("Unsupported file type. Only PDF and TXT are supported.")
            
            if not docs:
                raise Exception("No content could be extracted from the document")
            
            # Check if documents have content
            total_content = sum(len(doc.page_content) for doc in docs)
            if total_content == 0:
                raise Exception("Document appears to be empty or content could not be extracted")
            
            splits = self.text_splitter.split_documents(docs)
            print(f"Created {len(splits)} text chunks")
            
            if not splits:
                raise Exception("No text chunks were created from the document")
            
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.tools = [self.create_retriever_tool()]
            self.llm = self.llm.bind_tools(self.tools)
            # Rebuild the app with new tools
            self.app = self._build_graph()
            print("Document successfully ingested and vectorstore updated")
            
        except Exception as e:
            print(f"Error in ingest_document: {str(e)}")
            raise Exception(f"Error processing document: {str(e)}")

    def create_retriever_tool(self):
        if not self.vectorstore:
            raise Exception("No document loaded. Please upload a document or provide a default PDF.")
        
        # Load vector search settings from environment
        vector_search_k = int(os.getenv("VECTOR_SEARCH_K"))
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": vector_search_k})
        
        return create_retriever_tool(
            retriever,
            name="RAG_Doc_Search",
            description="""
            Search the uploaded document for relevant information to answer user questions.
            This tool will return the most relevant document passages based on semantic similarity.
            Use this tool when you need to find specific information from the uploaded document.
            The retrieved context should be used as the primary source for your responses.
            """
        )

    def prompt_with_context(self, context_chunks, user_question, confidence_score=None):
        """
        Enhanced prompt assembly with context management using config prompts
        """
        max_context_length = 3000  # Approximate token limit for context
        
        # Prepare context text
        if len(str(context_chunks)) > max_context_length:
            # Smart truncation - keep most relevant chunks
            context_summary = f"[Context truncated for length - showing top {len(context_chunks)} most relevant passages]\n\n"
            context_text = context_summary + "\n\n".join([chunk.page_content[:500] + "..." for chunk in context_chunks])
        else:
            context_text = "\n\n".join([chunk.page_content for chunk in context_chunks])
        
        # Add confidence indicator if available
        confidence_note = ""
        if confidence_score:
            confidence_note = f"\n**Retrieval Confidence:** {confidence_score:.2f}/1.0\n"
        
        # Use the context assembly prompt from config
        return config["rag_agent"]["context_assembly_prompt"].format(
            context=context_text + confidence_note,
            question=user_question
        )

    def classify_query(self, query):
        """Classify query type to optimize response strategy"""
        query_types = {
            "factual": ["what", "who", "when", "where", "how much", "how many"],
            "analytical": ["why", "how", "compare", "analyze", "explain"],
            "procedural": ["steps", "process", "procedure", "how to"],
            "summary": ["summarize", "overview", "main points", "key"]
        }
        
        query_lower = query.lower()
        for query_type, keywords in query_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        return "general"

    def chat_agent(self, state: RAGAgentState) -> RAGAgentState:
        if self.debug:
            print("chat_agent invoked with messages:")
            print("==== All State Messages ====")
            for msg in state["messages"]:
                print(f"- {type(msg).__name__}: {msg.content}") 
            print("==== All State Messages ====")

        if not self.vectorstore:
            system_prompt = SystemMessage(
                """
                You are a helpful assistant. However, no document has been uploaded yet.
                Please inform the user that they need to upload a document (PDF or TXT) to get started with document Q&A.
                Be polite and explain that once they upload a document, you'll be able to answer questions about it.
                """
            )
            response = self.llm.invoke([system_prompt] + state["messages"])
        else:
            last_human_message = None
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    last_human_message = message
                    break
            
            if last_human_message:
                try:
                    # Get context using enhanced prompting
                    vector_search_k = int(os.getenv("VECTOR_SEARCH_K"))
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": vector_search_k})
                    context_chunks = retriever.invoke(last_human_message.content)
                    
                    if context_chunks:
                        # Use enhanced prompt assembly
                        enhanced_context = self.prompt_with_context(context_chunks, last_human_message.content)
                        query_type = self.classify_query(last_human_message.content)
                        
                        # Create enhanced system prompt with context
                        system_prompt = SystemMessage(
                            config["rag_agent"]["system_prompt"] + 
                            f"\n\n**Current Query Type:** {query_type}\n\n" +
                            enhanced_context
                        )

                        if self.debug:
                            print("==== System Prompt with Context ====")
                            print("system_prompt :", system_prompt)

                        # Use conversation history but with enhanced context
                        conversation_messages = [system_prompt] + state["messages"][:-1] + [last_human_message]
                        response = self.llm.invoke(conversation_messages)
                    else:
                        # Fallback if no context found
                        system_prompt = SystemMessage(config["rag_agent"]["system_prompt"])
                        response = self.llm.invoke([system_prompt] + state["messages"])
                        
                except Exception as e:
                    print(f"Error in enhanced chat_agent: {str(e)}")
                    # Fallback to standard approach
                    system_prompt = SystemMessage(config["rag_agent"]["system_prompt"])
                    response = self.llm.invoke([system_prompt] + state["messages"])
            else:
                # No human message found, use standard approach
                system_prompt = SystemMessage(config["rag_agent"]["system_prompt"])
                response = self.llm.invoke([system_prompt] + state["messages"])
        
        return {"messages": [response]}

    def should_continue(self, state: RAGAgentState):
        last_message = state["messages"][-1]
        if not self.tools:  # No tools available, go straight to end
            return "end"
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def _build_graph(self):
        graph = StateGraph(RAGAgentState)
        graph.add_node("chat_agent", self.chat_agent)
        
        if self.tools:
            graph.add_node("tools", ToolNode(tools=self.tools))
            graph.set_entry_point("chat_agent")
            graph.add_conditional_edges(
                "chat_agent",
                self.should_continue,
                {"continue": "tools", "end": END}
            )
            graph.add_edge("tools", "chat_agent")
        else:
            # No tools available, simple chat flow
            graph.set_entry_point("chat_agent")
            graph.add_edge("chat_agent", END)
        
        return graph.compile(checkpointer=self.memory)

    def run(self, user_message: str, thread_id: str = None) -> str:
        """
        Main run method using LangGraph with enhanced prompting and conversation memory
        """
        # Use provided thread_id or generate a new one
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        config_dict = {"configurable": {"thread_id": thread_id}}
        
        # Get existing state from checkpoint or create new state
        # LangGraph's checkpointer will automatically merge with existing conversation
        try:
            existing_checkpoint = self.memory.get(config_dict)
            if existing_checkpoint and "channel_values" in existing_checkpoint:
                # Append to existing conversation
                existing_messages = existing_checkpoint["channel_values"].get("messages", [])

                if self.debug:
                    print(f"Existing messages found for thread_id {thread_id}:")
                    for msg in existing_messages:
                        print(f"- {type(msg).__name__}: {msg.content}")

                state = {"messages": existing_messages + [HumanMessage(content=user_message)]}
            else:
                # Start new conversation
                state = {"messages": [HumanMessage(content=user_message)]}
        except Exception as e:
            print(f"Warning: Could not retrieve existing checkpoint: {e}")
            # Fallback to new conversation
            state = {"messages": [HumanMessage(content=user_message)]}
        
        try:
            # Use LangGraph with enhanced prompting and conversation memory
            result = self.app.invoke(state, config=config_dict)
            
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content
            
            return "I couldn't generate a response."
        except Exception as e:
            print(f"Error in run method: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def load_internal_documents(self):
        """Load and index all documents from the data folder"""
        try:
            if not os.path.exists(self.data_folder):
                print(f"Data folder '{self.data_folder}' not found")
                return []
                
            # Find all PDF and TXT files in the data folder
            pdf_files = glob.glob(os.path.join(self.data_folder, "*.pdf"))
            txt_files = glob.glob(os.path.join(self.data_folder, "*.txt"))
            all_files = pdf_files + txt_files
            
            if not all_files:
                print(f"No PDF or TXT files found in '{self.data_folder}' folder")
                return []
            
            all_documents = []
            file_names = []
            
            for file_path in all_files:
                try:
                    file_name = os.path.basename(file_path)
                    print(f"Loading: {file_name}")
                    
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    else:  # txt file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        docs = [Document(page_content=content, metadata={"source": file_name})]
                    
                    if docs:
                        # Add source metadata to all documents
                        for doc in docs:
                            doc.metadata["source_file"] = file_name
                        all_documents.extend(docs)
                        file_names.append(file_name)
                        print(f"✓ Loaded {len(docs)} pages/sections from {file_name}")
                    
                except Exception as e:
                    print(f"✗ Error loading {file_name}: {str(e)}")
                    continue
            
            if all_documents:
                # Split all documents
                splits = self.text_splitter.split_documents(all_documents)
                print(f"Created {len(splits)} text chunks from {len(file_names)} files")
                
                # Create vectorstore
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                self.tools = [self.create_internal_retriever_tool(file_names)]
                self.llm = self.llm.bind_tools(self.tools)
                self.app = self._build_graph()
                
                return file_names
            else:
                print("No documents could be loaded from the data folder")
                return []
                
        except Exception as e:
            print(f"Error loading internal documents: {str(e)}")
            return []
    
    def create_internal_retriever_tool(self, file_names):
        """Create retriever tool for internal documents with enhanced prompting"""
        if not self.vectorstore:
            raise Exception("No documents loaded from data folder")
        
        vector_search_k = int(os.getenv("VECTOR_SEARCH_K", "5"))
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": vector_search_k})
        
        # Use the standard retriever tool but with enhanced description
        file_list = ", ".join(file_names)
        description = f"""
        Search the internal knowledge base for relevant information to answer user questions.
        
        Available documents: {file_list}
        
        This tool will return the most relevant passages from these internal documents based on semantic similarity.
        Use this tool when you need to find specific information from the company's internal knowledge base.
        The retrieved context should be used as the primary source for your responses, and you should mention
        which document(s) the information comes from when possible.
        """
        
        return create_retriever_tool(
            retriever,
            name="Internal_Knowledge_Search", 
            description=description
        )

    def get_available_internal_documents(self):
        """Get list of available internal documents"""
        if not os.path.exists(self.data_folder):
            return []
        
        pdf_files = glob.glob(os.path.join(self.data_folder, "*.pdf"))
        txt_files = glob.glob(os.path.join(self.data_folder, "*.txt"))
        all_files = pdf_files + txt_files
        
        return [os.path.basename(f) for f in all_files]

    def display_graph(self):
        """Display the LangGraph workflow diagram"""
        try:
            from IPython.display import Image, display
            display(Image(self.app.get_graph().draw_mermaid_png()))
        except ImportError:
            print("IPython not available. Use save_graph() to save the diagram to a file.")
        except Exception as e:
            print(f"Error displaying graph: {str(e)}")

    def save_graph(self, filename="rag_agent_graph.png"):
        """Save the LangGraph workflow diagram to a file"""
        try:
            graph_png = self.app.get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(graph_png)
            print(f"Graph saved as {filename}")
            return filename
        except Exception as e:
            print(f"Error saving graph: {str(e)}")
            return None

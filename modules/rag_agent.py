

import os
import glob
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
                        non_empty_docs.append(doc)
                        print(f"Page {i+1}: {len(content)} characters")
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
            This tool searches the uploaded document for relevant information to answer user questions.
            """
        )


    def chat_agent(self, state: RAGAgentState) -> RAGAgentState:
        if not self.vectorstore:
            system_prompt = SystemMessage(
                """
                You are a helpful assistant. However, no document has been uploaded yet.
                Please inform the user that they need to upload a document (PDF or TXT) to get started with document Q&A.
                Be polite and explain that once they upload a document, you'll be able to answer questions about it.
                """
            )
        else:
            system_prompt = SystemMessage(
                """
                You are a helpful assistant that answers questions using ONLY the provided document tool.
                If you do not know the answer, say so. Do not use your own memory.
                """
            )
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
        # Use provided thread_id or generate a new one
        import uuid
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": thread_id}}
        state = {"messages": [HumanMessage(content=user_message)]}
        
        try:
            # Use invoke for conversation with memory
            result = self.app.invoke(state, config=config)
            
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
        """Create retriever tool for internal documents"""
        if not self.vectorstore:
            raise Exception("No documents loaded from data folder")
        
        vector_search_k = int(os.getenv("VECTOR_SEARCH_K", "5"))
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": vector_search_k})
        
        file_list = ", ".join(file_names)
        description = f"""
        This tool searches the internal knowledge base containing the following documents: {file_list}.
        Use this tool to find relevant information from these internal documents to answer user questions.
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

#!/usr/bin/env python3
"""
Test script to display and save the RAG agent graph diagram
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from rag_agent import RAGAgent

def test_graph_display():
    """Test the graph display and save functionality"""
    
    # Initialize RAG agent
    print("Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Load internal documents to create tools
    print("Loading internal documents to activate tools...")
    loaded_docs = agent.load_internal_documents()
    if loaded_docs:
        print(f"✓ Loaded {len(loaded_docs)} internal documents: {', '.join(loaded_docs)}")
    else:
        print("No internal documents found. Graph will show basic flow without tools.")
    
    # Save the graph diagram
    print("\nSaving graph diagram...")
    saved_file = agent.save_graph("rag_agent_langgraph.png")
    
    if saved_file:
        print(f"✓ Graph saved successfully as {saved_file}")
    else:
        print("✗ Failed to save graph")
    
    # Try to display (if in Jupyter/IPython environment)
    print("\nAttempting to display graph...")
    agent.display_graph()
    
    print("\nDone! You can now:")
    print("1. Use agent.display_graph() in Jupyter notebooks")
    print("2. Use agent.save_graph('filename.png') to save diagrams")

if __name__ == "__main__":
    test_graph_display()
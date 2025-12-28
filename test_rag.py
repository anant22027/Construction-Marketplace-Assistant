"""
Test script for the RAG pipeline
Tests various queries and displays results
"""

import os
from rag_pipeline import RAGPipeline

def test_queries():
    """Test the RAG pipeline with sample queries"""
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(openrouter_api_key=api_key)
    
    # Try to load existing index
    if os.path.exists("construction_index.index"):
        print("Loading existing index...")
        rag.load_index("construction_index")
    else:
        print("Building new index...")
        documents = rag.load_documents("documents")
        chunks = rag.chunk_documents(documents)
        rag.build_index(chunks)
        rag.save_index("construction_index")
    
    # Test queries
    test_questions = [
        "What factors affect construction project delays?",
        "What safety equipment is mandatory on construction sites?",
        "What is the minimum compressive strength for foundation concrete?",
        "How long does a typical residential construction project take?",
        "What permits are required for construction projects?",
        "How are change orders handled during construction?",
    ]
    
    print("\n" + "="*80)
    print("TESTING RAG PIPELINE")
    print("="*80)
    
    for i, query in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        
        result = rag.query(query, top_k=3)
        
        print("RETRIEVED CHUNKS:")
        print("-" * 80)
        for j, (chunk, metadata, distance) in enumerate(result['retrieved_chunks'], 1):
            print(f"\nChunk {j}:")
            print(f"  Source: {metadata['source']}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Content: {chunk[:200]}..." if len(chunk) > 200 else f"  Content: {chunk}")
        
        print("\n" + "-" * 80)
        print("GENERATED ANSWER:")
        print("-" * 80)
        print(result['answer'])
        print("\n")
    
    print("="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_queries()


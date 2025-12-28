"""
RAG Pipeline for Construction Marketplace Assistant
Implements document chunking, embedding, vector search, and LLM-based answer generation
"""

import os
import pickle
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import json


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 openrouter_api_key: str = None,
                 openrouter_model: str = "meta-llama/llama-3.2-3b-instruct:free"):
        """
        Initialize the RAG pipeline
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            openrouter_api_key: API key for OpenRouter (if None, will try to get from env)
            openrouter_model: Model name for OpenRouter
        """
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize OpenRouter client
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_api_key:
            self.llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
            self.llm_model = openrouter_model
        else:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        # Initialize storage
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.index = None
        self.is_indexed = False
    
    def load_documents(self, document_dir: str = "documents") -> List[Tuple[str, str]]:
        """
        Load all text documents from a directory
        
        Args:
            document_dir: Directory containing text documents
            
        Returns:
            List of (filename, content) tuples
        """
        documents = []
        if not os.path.exists(document_dir):
            raise FileNotFoundError(f"Document directory not found: {document_dir}")
        
        for filename in os.listdir(document_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(document_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append((filename, content))
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Tuple[str, str]], 
                       chunk_size: int = 500, 
                       chunk_overlap: int = 100) -> List[Dict]:
        """
        Chunk documents into smaller segments
        
        Args:
            documents: List of (filename, content) tuples
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        all_chunks = []
        
        for filename, content in documents:
            # Split by paragraphs first for better semantic coherence
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If adding this paragraph would exceed chunk size, save current chunk
                if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                    all_chunks.append({
                        'text': current_chunk.strip(),
                        'source': filename,
                        'chunk_index': chunk_index
                    })
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                    chunk_index += 1
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add the last chunk
            if current_chunk.strip():
                all_chunks.append({
                    'text': current_chunk.strip(),
                    'source': filename,
                    'chunk_index': chunk_index
                })
        
        print(f"Created {len(all_chunks)} chunks from documents")
        return all_chunks
    
    def build_index(self, chunks: List[Dict]):
        """
        Build FAISS index from document chunks
        
        Args:
            chunks: List of chunk dictionaries
        """
        self.chunks = [chunk['text'] for chunk in chunks]
        self.chunk_metadata = chunks
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.index.add(embeddings)
        
        self.is_indexed = True
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, Dict, float]]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk_text, metadata, distance) tuples
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve chunks and metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk_text = self.chunks[idx]
                metadata = self.chunk_metadata[idx]
                distance = float(distances[0][i])
                results.append((chunk_text, metadata, distance))
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, Dict, float]]) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User query
            retrieved_chunks: List of (chunk_text, metadata, distance) tuples
            
        Returns:
            Generated answer string
        """
        # Prepare context from retrieved chunks
        context_parts = []
        for i, (chunk_text, metadata, distance) in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source: {metadata['source']}]\n{chunk_text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt with strict grounding instructions
        prompt = f"""You are a helpful assistant for a construction marketplace. Answer the user's question using ONLY the information provided in the context below. 

IMPORTANT RULES:
- Base your answer STRICTLY on the provided context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question completely."
- Do NOT use any external knowledge or make assumptions beyond what's in the context
- Be clear and concise
- Cite which source document your information comes from when relevant

CONTEXT:
{context}

USER QUESTION: {query}

ANSWER (based only on the context above):"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based strictly on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more grounded responses
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "No endpoints found" in error_str:
                return f"Error: The model '{self.llm_model}' is not available. Please try a different model from the sidebar (Llama 3.2 3B is recommended)."
            else:
                return f"Error generating answer: {error_str}"
    
    def query(self, user_query: str, top_k: int = 3) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate
        
        Args:
            user_query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with query, retrieved_chunks, and answer
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(user_query, top_k)
        
        # Generate answer
        answer = self.generate_answer(user_query, retrieved_chunks)
        
        return {
            'query': user_query,
            'retrieved_chunks': retrieved_chunks,
            'answer': answer
        }
    
    def save_index(self, filepath: str):
        """Save the FAISS index and chunks to disk"""
        if not self.is_indexed:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, f"{filepath}.index")
        with open(f"{filepath}.chunks", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f)
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load the FAISS index and chunks from disk"""
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.chunks", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['metadata']
        self.is_indexed = True
        print(f"Index loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it using: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Load and process documents
    documents = rag.load_documents("documents")
    chunks = rag.chunk_documents(documents)
    
    # Build index
    rag.build_index(chunks)
    
    # Save index for future use
    rag.save_index("construction_index")
    
    # Example query
    print("\n" + "="*50)
    print("Example Query:")
    query = "What factors affect construction project delays?"
    result = rag.query(query, top_k=3)
    
    print(f"\nQuery: {result['query']}")
    print("\nRetrieved Chunks:")
    for i, (chunk, metadata, distance) in enumerate(result['retrieved_chunks'], 1):
        print(f"\n--- Chunk {i} (Source: {metadata['source']}, Distance: {distance:.4f}) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"\n\nAnswer:\n{result['answer']}")


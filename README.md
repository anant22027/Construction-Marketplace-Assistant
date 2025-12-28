# Construction Marketplace RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about construction projects using internal documents (policies, FAQs, and specifications).

## Features

- **Document Processing**: Chunks documents into meaningful segments
- **Semantic Search**: Uses FAISS for efficient vector similarity search
- **Grounded Answers**: LLM generates answers strictly from retrieved context
- **Transparency**: Displays retrieved chunks and generated answers
- **Interactive Interface**: Streamlit-based chatbot for easy interaction

## Architecture

### Components

1. **Document Chunking**: Documents are split into overlapping chunks (500 chars, 100 char overlap) to preserve context
2. **Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` for generating 384-dimensional embeddings
3. **Vector Index**: FAISS (Flat L2) index for fast similarity search
4. **Retrieval**: Top-k most similar chunks retrieved using cosine similarity (via L2 distance)
5. **Answer Generation**: OpenRouter LLM generates answers with strict grounding instructions

## Setup

### Prerequisites

- Python 3.8+
- OpenRouter API key (free tier available at https://openrouter.ai)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd "indecimal rag"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenRouter API key:
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your-api-key-here"

# Linux/Mac
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file (not included in repo):
```
OPENROUTER_API_KEY=your-api-key-here
```

## Usage

### Option 1: Streamlit Web Interface (Recommended)

Launch the interactive chatbot:
```bash
streamlit run app.py
```

Then:
1. Enter your OpenRouter API key in the sidebar
2. Select an LLM model
3. Click "Initialize/Reload RAG Pipeline"
4. Start asking questions!

### Option 2: Command Line

Run the pipeline directly:
```bash
python rag_pipeline.py
```

## Model Choices

### Embedding Model: `all-MiniLM-L6-v2`
- **Why**: Lightweight (80MB), fast, and provides good semantic understanding
- **Dimensions**: 384
- **Performance**: Good balance between quality and speed for this use case

### LLM: OpenRouter Free Models
- **Primary**: `meta-llama/llama-3.2-3b-instruct:free` (default, most reliable)
- **Alternatives**: 
  - `mistralai/mistral-7b-instruct:free`
  - `google/gemini-flash-1.5:free`
- **Why**: Free tier available, good performance, easy API integration
- **Note**: Some model endpoints may not be available. If you get a 404 error, try switching to a different model in the Streamlit sidebar.

## Document Chunking and Retrieval

### Chunking Strategy
- **Method**: Paragraph-based chunking with overlap
- **Chunk Size**: 500 characters (configurable)
- **Overlap**: 100 characters to preserve context across boundaries
- **Rationale**: Preserves semantic coherence while allowing fine-grained retrieval

### Retrieval Implementation
- **Vector Store**: FAISS IndexFlatL2 (exact L2 distance search)
- **Similarity Metric**: L2 distance (Euclidean), converted to similarity score
- **Top-k**: Default 3 chunks, configurable in UI
- **Process**: 
  1. Query embedded using same model as documents
  2. FAISS searches for nearest neighbors
  3. Top-k chunks returned with metadata (source, distance)

## Grounding to Retrieved Context

The system enforces grounding through:

1. **Explicit Prompt Instructions**: LLM is instructed to answer ONLY from provided context
2. **Context Formatting**: Retrieved chunks are clearly marked with source documents
3. **Temperature Control**: Low temperature (0.3) reduces hallucination
4. **Fallback Handling**: If context is insufficient, LLM is instructed to say so
5. **Transparency**: All retrieved chunks are displayed to users

### Prompt Template
```
You are a helpful assistant for a construction marketplace. Answer the user's question using ONLY the information provided in the context below. 

IMPORTANT RULES:
- Base your answer STRICTLY on the provided context
- If the context doesn't contain enough information, say "I don't have enough information..."
- Do NOT use any external knowledge or make assumptions beyond what's in the context
- Be clear and concise
- Cite which source document your information comes from when relevant

CONTEXT:
[Retrieved chunks with source attribution]

USER QUESTION: [user query]

ANSWER (based only on the context above):
```

## Transparency and Explainability

The system provides full transparency:

1. **Retrieved Context Display**: Shows all chunks used for answering
2. **Source Attribution**: Each chunk shows its source document
3. **Similarity Scores**: Distance/similarity metrics displayed
4. **Answer Visibility**: Generated answer clearly separated from context

## Project Structure

```
.
├── documents/                    # Source documents
│   ├── construction_policies.txt
│   ├── construction_faqs.txt
│   └── construction_specifications.txt
├── rag_pipeline.py              # Core RAG implementation
├── app.py                       # Streamlit interface
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── construction_index.index     # Generated FAISS index (after first run)
└── construction_index.chunks    # Generated chunk metadata (after first run)
```

## Example Queries

- "What factors affect construction project delays?"
- "What safety protocols must be followed on construction sites?"
- "How long does a typical residential construction project take?"
- "What permits are required for construction projects?"
- "What are the concrete specifications for foundations?"
- "How do I handle change orders during construction?"

## Limitations and Future Improvements

### Current Limitations
1. **Document Format**: Currently supports only `.txt` files
2. **Chunking**: Fixed-size chunking may split related information
3. **No Re-ranking**: Simple top-k retrieval without re-ranking
4. **Single Language**: English only

### Potential Enhancements
1. **Multi-format Support**: PDF, DOCX parsing
2. **Advanced Chunking**: Semantic chunking using embeddings
3. **Re-ranking**: Cross-encoder re-ranking for better relevance
4. **Local LLM**: Integration with Ollama for offline use
5. **Evaluation Framework**: Automated quality metrics
6. **Query Expansion**: Improve retrieval with query reformulation

## Evaluation

To evaluate the system, consider:

1. **Relevance**: Are retrieved chunks relevant to the query?
2. **Completeness**: Does the answer cover all aspects of the question?
3. **Groundedness**: Is the answer supported by retrieved context?
4. **Clarity**: Is the answer clear and well-structured?

### Sample Test Questions

1. What factors affect construction project delays?
2. What is the minimum compressive strength for foundation concrete?
3. How are change orders handled during construction?
4. What safety equipment is mandatory on construction sites?
5. What is the typical timeline for residential construction?
6. What permits are required for construction projects?
7. What are the insulation R-value requirements?
8. What electrical standards must be followed?

## License

This project is provided as-is for evaluation purposes.

## Contact

For questions or issues, please refer to the repository issues page.


# RAG Pipeline - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Component Guide](#component-guide)
4. [Setup & Installation](#setup--installation)
5. [Usage Guide](#usage-guide)
6. [Data Flow](#data-flow)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is This?

A **Retrieval-Augmented Generation (RAG) system** designed to answer questions about research papers. It combines:
- **Dense semantic search** (embeddings + FAISS)
- **Sparse keyword search** (BM25)
- **Query expansion** (synonyms + pseudo-relevance feedback)
- **Section-aware ranking** (document structure awareness)
- **Cross-encoder reranking** (deep relevance scoring)
- **LLM generation** (Groq's llama-3.3-70b)

### Key Problem Solved

Traditional search engines struggle with:
- Vocabulary mismatch (query uses different terms than documents)
- Section blindness (not knowing which part of a document to prioritize)
- Noise in results (figure captions, tables, references)
- Low diversity in ranking

This system addresses all four through intelligent retrieval orchestration.

---

## System Architecture

### High-Level Pipeline

```
User Question
    ↓
Query Expansion (synonyms) → "training" → ["reward", "PPO", "policy learning", ...]
    ↓
Dense Search (FAISS) + BM25 Search (Parallel)
    ↓
Pseudo-Relevance Feedback (Analyze top results for refinement)
    ↓
Re-retrieval with expanded query
    ↓
Section Prior Injection (Ensure relevant sections are included)
    ↓
Noise Filtering (Remove figures, tables, references)
    ↓
Hybrid Fusion (Combine dense + sparse with normalization)
    ↓
Cross-Encoder Reranking (Deep relevance scoring)
    ↓
Section Boosting (Final score adjustment for document structure)
    ↓
Context Building & LLM Generation
    ↓
Final Answer with Citations
```

### Module Dependency Graph

```
loader.py
    ↓
chunker.py
    ↓
embedder.py
    ├─ FAISS index
    ├─ BM25 index
    └─ Chunk store
    ↓
retriever.py ⭐ Core
    ├─ query_expander.py
    ├─ prf_expander.py
    └─ section_prior.py
    ↓
generator.py (RAG endpoint)
    ↓
evaluate.py, failure_analysis.py (Evaluation)
```

---

## Component Guide

### 1. Document Loader (`loader.py`)

**Responsibility**: Extract and clean text from research papers

```python
from loader import ResearchPaperLoader

# Initialize
loader = ResearchPaperLoader("data/sample.pdf")

# Load pages
pages = loader.load()
# Output: [
#   {"text": "Introduction...", "page": 0, "paper": "sample.pdf"},
#   {"text": "Related Work...", "page": 1, "paper": "sample.pdf"},
#   ...
# ]
```

**Features**:
- Extracts text per page using PyMuPDF
- Cleans: removes extra whitespace, collapses newlines
- Filters out References section
- Skips empty/minimal pages

**Configuration**: None (straightforward)

---

### 2. Section Chunker (`chunker.py`)

**Responsibility**: Split documents into semantic chunks with section tracking

```python
from chunker import SectionChunker

chunker = SectionChunker(
    chunk_size=200,        # Max words per chunk
    overlap=30,            # Word overlap between chunks
    min_chunk_words=30     # Minimum words in a chunk
)

chunks = chunker.chunk_documents(pages)
# Output: [
#   {
#     "text": "Introduction discusses the problem of ...",
#     "section": "1. introduction",
#     "paper": "sample.pdf"
#   },
#   ...
# ]
```

**Features**:
- Detects section headings with regex patterns
- Maintains section context for each chunk
- Merges small paragraphs intelligently
- Handles chunk overlap to preserve context
- Enforces minimum chunk size

**Tuning Parameters**:
- `chunk_size`: Increase for broader context (but fewer chunks)
- `overlap`: Increase to preserve more context between chunks
- `min_chunk_words`: Filter out very small chunks

---

### 3. Embedder (`embedder.py`)

**Responsibility**: Build search indices (dense + sparse)

```python
from embedder import ResearchPaperIndexer

indexer = ResearchPaperIndexer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_hnsw=False  # True for 100K+ chunks (approximate search)
)

# Build dense index (FAISS)
dense_index, embeddings = indexer.build_dense_index(chunks)

# Build sparse index (BM25)
bm25 = indexer.build_bm25_index(chunks)

# Save all
indexer.save_all(dense_index, bm25, chunks)
```

**Features**:
- **Dense**: Sentence Transformers → normalized embeddings → FAISS
- **Sparse**: BM25 (Okapi) for keyword matching
- Batch processing with progress tracking
- Supports HNSW for very large corpora

**Output Files**:
- `faiss_index.bin`: Dense vector index
- `bm25.pkl`: Sparse retrieval model
- `chunks.pkl`: Original chunk data

---

### 4. Query Expander (`query_expander.py`)

**Responsibility**: Expand queries with domain-specific synonyms

```python
from query_expander import expand_query

original = "How is training encouraged?"
expanded = expand_query(original)
# Output: "How is training encouraged? reward shaping penalty incentive bonus..."
```

**How It Works**:
1. Checks if query contains known keywords (training, reward, dataset, etc.)
2. Appends associated synonyms
3. Creates a richer query for retrieval

**Customization**: Edit `RESEARCH_SYNONYMS` dict in the file to add domain terms

---

### 5. Pseudo-Relevance Feedback (`prf_expander.py`)

**Responsibility**: Refine queries based on initial retrieval

```python
from prf_expander import expand_query_prf

initial_chunks = [...]  # Top-10 initial results
expanded = expand_query_prf("original query", initial_chunks)
# Output: "original query term1 term2 term3 ..."
```

**How It Works**:
1. Extracts all words from top-10 results
2. Filters stopwords and short terms
3. Counts frequency
4. Appends top-5 most frequent terms to query

**Purpose**: Overcome vocabulary mismatch by learning from relevant documents

---

### 6. Retriever (`retriever.py`) ⭐ Core Component

**Responsibility**: Multi-stage intelligent retrieval pipeline

```python
from retriever import ResearchPaperRetriever

retriever = ResearchPaperRetriever(
    bi_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    dense_top_k=30,
    bm25_top_k=30,
    rerank_top_k=20,
    final_top_k=5
)

results = retriever.search("What is the reward function?", top_k=5)
# Output: [
#   {
#     "text": "The reward function is defined as...",
#     "section": "3.1 reward function",
#     "paper": "sample.pdf",
#     "score": 0.8542
#   },
#   ...
# ]
```

**Pipeline Stages**:

1. **Query Expansion** (query_expander.py)
   - Add synonyms to the query

2. **Initial Dense Search** (FAISS + bi-encoder)
   - Retrieve top-30 candidates via cosine similarity
   - Normalized embeddings for stable scores

3. **PRF Analysis** (prf_expander.py)
   - Extract frequent terms from top-10 results
   - Build expanded query

4. **Re-retrieval** (FAISS + expanded query)
   - Search again with richer query
   - Get wider candidate pool

5. **Section Prior Injection** (section_prior.py)
   - Analyze query intent
   - Inject chunks from matching document sections
   - Solves embedding blindness

6. **Noise Filtering**
   - Remove figure captions (detected via regex)
   - Filter high non-ASCII chunks (equations)
   - Remove reference entries and tables

7. **Cross-Encoder Reranking** (ms-marco)
   - Score top-20 (query, chunk) pairs
   - Deep relevance assessment

8. **Section Boosting** (section_prior.py)
   - Add +8.0 score bonus to chunks matching query intent sections
   - Re-rank final results

9. **Final Selection**
   - Return top-5 highest-scoring chunks

**Tuning Parameters**:
- `dense_top_k`: Increase to get more candidates (slower but broader)
- `rerank_top_k`: Increase to rerank more candidates (more accurate but slower)
- `final_top_k`: Results returned to user

---

### 7. Section Prior (`section_prior.py`)

**Responsibility**: Map query intent to document sections

```python
from section_prior import get_section_prior, compute_section_boost

# Detect query intent
sections = get_section_prior("How is training encouraged?")
# Output: ["reward", "3.1"]

# Compute boost for a chunk
boost = compute_section_boost("3.1 reward function", ["reward", "3.1"])
# Output: 8.0 (matched!) or 0.0 (no match)
```

**Built-in Patterns** (15+ rules):
- Reward/training queries → "reward", "3.1"
- Obstacle avoidance → "obstacle avoidance", "2.4"
- Dataset queries → "dataset", "image", "4.2"
- Experimental setup → "experiment", "setup", "4."
- Conclusion queries → "conclusion", "6."
- ... and more

**Benefit**: Addresses embedding blindness where embeddings fail to recognize query-section relevance

---

### 8. Generator (`generator.py`) - RAG Endpoint

**Responsibility**: Generate answers from retrieved context

```python
from generator import ResearchPaperRAG

rag = ResearchPaperRAG()

answer = rag.answer("What is the reward function design?")
# Output: "The reward function comprises three components..."
```

**How It Works**:
1. Retrieves top-10 chunks via retriever.search()
2. Formats top-4 chunks as context blocks
3. Builds prompt: "Answer ONLY using provided context"
4. Calls Groq LLM (llama-3.3-70b-versatile)
5. Returns generated answer

**Anti-Hallucination Measures**:
```
Rules:
- Do NOT hallucinate
- If answer not present → say "Not found in paper"
- Always cite chunk number and section
- Be concise but informative
- Merge ideas from multiple chunks
```

**Configuration**:
- Model: llama-3.3-70b-versatile
- Temperature: 0.1 (low randomness)
- Max tokens: 600

---

### 9. Evaluator (`evaluate.py`)

**Responsibility**: Measure retrieval quality

```python
from evaluate import evaluate

# Run evaluation on dataset
evaluate()
# Prints: Recall@5, MRR, Hit Rate
```

**Metrics**:
- **Recall@5**: % of queries with relevant chunk in top-5
- **MRR**: Mean reciprocal rank of first relevant chunk
- **Hit Rate**: Overall success rate

**Evaluation Method**: Semantic similarity (embeddings) with 0.4 threshold

---

### 10. Failure Analyzer (`failure_analysis.py`)

**Responsibility**: Debug retrieval failures

```python
from failure_analysis import analyze

# Analyze failed queries
analyze()
# Prints: Failed queries with expected vs retrieved sections
```

**Output**: Insights into what went wrong (wrong section, noise, etc.)

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/ZenMachina16/rag_pipeline.git
cd rag_pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with API key
echo "GROQ_API_KEY=your_key_here" > src/.env

# 4. Prepare data directory
mkdir -p data
# Place your PDF research papers in data/
```

### Required Dependencies

```
pymupdf              # PDF processing
faiss-cpu           # Dense search (use faiss-gpu for NVIDIA GPU)
sentence-transformers  # Embeddings
rank-bm25           # Sparse search
groq                # LLM generation
python-dotenv       # Config management
numpy               # Numerical ops
```

### Groq API Setup

1. Sign up at https://console.groq.com
2. Create API key
3. Save to `src/.env`:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
   ```

---

## Usage Guide

### Workflow 1: Index a New Paper

```bash
cd src

# Step 1: Load PDF
python -c "
from loader import ResearchPaperLoader
loader = ResearchPaperLoader('../data/my_paper.pdf')
pages = loader.load()
print(f'Loaded {len(pages)} pages')
"

# Step 2: Chunk the paper
python chunker.py

# Step 3: Build indices
python embedder.py

# (chunks.pkl, faiss_index.bin, bm25.pkl are now ready)
```

### Workflow 2: Ask Questions (RAG)

```bash
cd src

# Interactive Q&A
python generator.py

# Example:
# Ask Question: What is the reward function?
# ================ ANSWER ================
# The reward function comprises three components...
```

### Workflow 3: Evaluate on Benchmark

```bash
cd src

# (Requires data/eval_dataset_v2.json)
python evaluate.py

# Output:
# Queries: 20
# Recall@5: 0.850
# MRR: 0.780
# Hit Rate: 0.850
```

### Workflow 4: Debug Failures

```bash
cd src

# Analyze failed queries
python failure_analysis.py

# Output:
# ❌ FAILED QUERY: What is the dynamic obstacle scenario?
# Expected Sections: ['dynamic obstacle', '4. dynamic']
# Expected Keywords: ['dynamic', 'obstacle', 'scenario']
# 
# Top Retrieved Chunks:
# Rank: 1
# Score: 0.5234
# Section: 3.1 reward function
# Text: The reward function...
```

---

## Data Flow

### Indexing Phase (One-time)

```
my_paper.pdf
    ↓
[loader.py] → Extract text from pages
    ↓
pages = [
  {text: "...", page: 0, paper: "my_paper.pdf"},
  ...
]
    ↓
[chunker.py] → Split into semantic chunks
    ↓
chunks = [
  {text: "...", section: "introduction", paper: "my_paper.pdf"},
  ...
]
    ↓
[embedder.py] → Create indices
    ├─ Dense: Embed chunks → FAISS
    ├─ Sparse: BM25
    └─ Store: chunks.pkl
    ↓
Ready for retrieval!
```

### Retrieval Phase (Per Query)

```
User Query: "What is the reward function?"
    ↓
[expand_query] → "What is... + reward shaping penalty incentive bonus"
    ↓
[dense_search via FAISS] → Top-30 candidates
    ↓
[prf_expand based on top-10] → "What is... + reward function trajectory smoothing"
    ↓
[dense_search again with expanded] → Wider pool
    ↓
[section_prior] → Inject chunks from "reward" & "3.1" sections
    ↓
[noise_filter] → Remove figures, references
    ↓
[cross_encoder_rerank] → Deep scoring
    ↓
[section_boost] → +8.0 for matching sections
    ↓
Return top-5 ranked chunks
```

### Generation Phase

```
Top-5 chunks
    ↓
[build_context] → Format as "[Chunk 1 | Section X]\n..."
    ↓
[build_prompt] → "You are assistant. Answer ONLY using context. Context: [...]\nQuestion: [...]"
    ↓
[LLM call] → Groq llama-3.3-70b
    ↓
Answer: "The reward function comprises..."
```

---

## API Reference

### ResearchPaperLoader

```python
class ResearchPaperLoader:
    def __init__(self, pdf_path: str)
    def load() -> List[Dict]  # Returns pages with text
    def clean_text(text: str) -> str
```

### SectionChunker

```python
class SectionChunker:
    def __init__(self, chunk_size=200, overlap=30, min_chunk_words=30)
    def chunk_documents(pages: List[Dict]) -> List[Dict]
    def extract_section_paragraphs(pages) -> List[Dict]
    def build_chunks(section_paragraphs, page_map) -> List[Dict]
```

### ResearchPaperIndexer

```python
class ResearchPaperIndexer:
    def __init__(self, model_name: str, use_hnsw: bool)
    def build_dense_index(chunks) -> (index, embeddings)
    def build_bm25_index(chunks) -> bm25_model
    def save_all(index, bm25, chunks) -> None
```

### ResearchPaperRetriever ⭐

```python
class ResearchPaperRetriever:
    def __init__(self, bi_encoder_name: str, cross_encoder_name: str, 
                 dense_top_k: int, bm25_top_k: int, rerank_top_k: int, 
                 final_top_k: int)
    
    def search(query: str, top_k: int = 5) -> List[Dict]
    
    # Private methods (advanced)
    def _dense_search(query) -> Dict[int, float]
    def _bm25_search(query) -> Dict[int, float]
    def _hybrid_fusion(dense_scores, bm25_scores) -> List[Tuple[int, float]]
    def _rerank(query, candidates) -> List[Tuple[int, float]]
```

### ResearchPaperRAG

```python
class ResearchPaperRAG:
    def __init__()
    def answer(query: str) -> str  # Main API
    def build_context(results, max_chunks=4) -> str
    def build_prompt(query, context) -> str
```

### Query Expansion Functions

```python
def expand_query(query: str) -> str
def expand_query_prf(query: str, initial_chunks: List[Dict]) -> str
def get_section_prior(query: str) -> List[str]
def compute_section_boost(chunk_section: str, preferred_sections: List[str]) -> float
```

---

## Configuration

### Global Parameters to Tune

#### Chunking
```python
# In chunker.py
chunk_size = 200        # Increase for broader chunks
overlap = 30            # Increase for more context preservation
min_chunk_words = 30    # Filter tiny chunks
```

#### Embedding
```python
# In embedder.py
model_name = "sentence-transformers/all-MiniLM-L6-v2"
use_hnsw = False        # Set True for 100K+ chunks
batch_size = 32         # Increase for faster embedding
```

#### Retrieval
```python
# In retriever.py initialization
dense_top_k = 30        # Candidates from dense search
bm25_top_k = 30         # Candidates from BM25
rerank_top_k = 20       # Candidates for cross-encoder
final_top_k = 5         # Final results
dense_weight = 0.6      # Weight for dense scores
bm25_weight = 0.4       # Weight for BM25 scores
```

#### Section Boost
```python
# In section_prior.py
section_boost = 8.0     # Score boost for matching sections
```

#### Generation
```python
# In generator.py
model = "llama-3.3-70b-versatile"
temperature = 0.1       # Lower = more deterministic
max_tokens = 600        # Max output length
max_chunks = 4          # Max context chunks
```

---

## Performance Tuning

### Speed Optimization

**For faster queries (sacrifice accuracy)**:
```python
# In retriever.py
rerank_top_k = 10       # Rerank fewer candidates
final_top_k = 3         # Return fewer results
```

**Enable HNSW for large corpus**:
```python
indexer = ResearchPaperIndexer(use_hnsw=True)
# ~10-50x faster but approximate (99% recall typical)
```

### Accuracy Optimization

**For better results (slower)**:
```python
# In retriever.py
rerank_top_k = 50       # Rerank more candidates
dense_top_k = 50        # Retrieve more dense candidates
```

**Increase query expansion**:
```python
# In prf_expander.py
top_n = 10              # Extract more PRF terms
```

### Memory Optimization

**For limited RAM** (e.g., Raspberry Pi):
```python
# Use smaller embeddings
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# Reduce embedding dimension via quantization
# Or use HNSW with reduced memory via IndexHNSWSQ
```

---

## Troubleshooting

### Issue: "GROQ_API_KEY not found"

**Solution**: Create `src/.env` file:
```bash
echo "GROQ_API_KEY=your_key_here" > src/.env
```

### Issue: "FileNotFoundError: chunks.pkl"

**Solution**: Run the indexing pipeline first:
```bash
cd src
python chunker.py
python embedder.py
```

### Issue: "Very low recall (< 50%)"

**Diagnoses**:
1. Check if chunk_size is too small (increase to 300)
2. Run failure_analysis.py to see patterns
3. Verify section_prior rules match your paper structure
4. Increase dense_top_k and rerank_top_k

**Fix**:
```python
# In retriever.py
dense_top_k = 50       # Was 30
rerank_top_k = 30      # Was 20

# In chunker.py
chunk_size = 300       # Was 200
overlap = 50           # Was 30
```

### Issue: "Slow queries (> 10 seconds)"

**Diagnoses**:
1. Network latency to Groq API
2. Cross-encoder reranking is slow
3. Too many chunks being reranked

**Fix**:
```python
# Reduce reranking pool
rerank_top_k = 10      # Was 20

# Or use --no-rerank flag (skip cross-encoder entirely)
```

### Issue: "Figure captions in top results"

**Root Cause**: Noise filtering threshold too low

**Fix** (in retriever.py):
```python
# Increase minimum chunk length
if len(text.split()) < 50:  # Was 20
    return True

# Increase ASCII ratio threshold
if non_ascii_ratio > 0.25:  # Was 0.35
    return True
```

### Issue: "Wrong sections retrieved"

**Root Cause**: Section Prior rules don't match your paper

**Fix**: Edit SECTION_PRIOR_RULES in section_prior.py to match:
```python
SECTION_PRIOR_RULES = [
    (r'your_query_pattern', ["your_section_1", "your_section_2"]),
    ...
]
```

---

## Example Workflows

### Example 1: Simple Q&A

```bash
cd src
python generator.py

# Enter queries interactively
```

### Example 2: Batch Evaluation

```python
from retriever import ResearchPaperRetriever
import json

retriever = ResearchPaperRetriever()

with open("test_queries.json") as f:
    queries = json.load(f)

for q in queries:
    results = retriever.search(q, top_k=5)
    print(f"Query: {q}")
    for r in results:
        print(f"  - {r['section']}: {r['text'][:100]}")
```

### Example 3: Custom Evaluation Metrics

```python
from retriever import ResearchPaperRetriever

retriever = ResearchPaperRetriever()

# NDCG@5
results = retriever.search("sample query", top_k=5)
relevances = [1, 0.8, 0.5, 0, 0]  # Custom relevance scores

dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevances))
# ... compute NDCG
```

---

## Advanced Customization

### Add Custom Synonyms

Edit `src/query_expander.py`:
```python
RESEARCH_SYNONYMS = {
    "your_term": ["synonym1", "synonym2", ...],
    ...
}
```

### Add Custom Section Rules

Edit `src/section_prior.py`:
```python
SECTION_PRIOR_RULES = [
    (r'your_query_regex', ["section_substring", "..."]),
    ...
]
```

### Replace LLM

Replace in `src/generator.py`:
```python
# Instead of Groq
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    ...
)
```

### Replace Embedder

Replace in `src/embedder.py`:
```python
# Instead of Sentence Transformers
from transformers import AutoModel
self.model = AutoModel.from_pretrained("your-model")
```

---

## Summary

This is a production-ready RAG system for research papers with:
- ✅ Multi-stage intelligent retrieval
- ✅ Hybrid dense + sparse search
- ✅ Query expansion (synonyms + PRF)
- ✅ Section-aware ranking
- ✅ Noise filtering
- ✅ Cross-encoder reranking
- ✅ LLM generation with anti-hallucination
- ✅ Evaluation & failure analysis tools
- ✅ Modular, extensible architecture

Start with the indexing workflow, then use the RAG endpoint for interactive Q&A!

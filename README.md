# Vibe Matcher: Semantic Fashion Recommendation Prototype

Vibe Matcher is a semantic recommendation system prototype that matches fashion products to a user's vibe-based query, such as *"edgy streetwear vibe."*  
It demonstrates how vector-based semantic understanding can capture a user’s intended mood or style, rather than relying solely on keyword overlap.  
The system leverages **text embeddings**, **vector search**, and **language model reasoning** to identify fashion items most aligned with the user’s described vibe.

## Project Overview

Traditional search systems rely heavily on exact keywords and often fail to interpret stylistic or emotional intent.  
Vibe Matcher addresses this limitation by converting product descriptions and user queries into dense numerical vectors.  
These vectors are compared using **cosine similarity** and **semantic retrieval**, enabling the system to recommend conceptually relevant fashion items even when the wording differs.

## Implementation Details

The original design of this prototype intended to use **OpenAI’s `text-embedding-ada-002` model** for text vectorization.  
However, due to the expiration of the associated API key, the system was transitioned to  
**Hugging Face’s `SentenceTransformer` model (`all-MiniLM-L6-v2`)**, which provides comparable performance for semantic similarity tasks.

This change ensures:
- Reproducibility in restricted or resource-limited environments  
- Consistent embedding quality suitable for experimentation and evaluation  

The substitution does not alter the overall methodology or outcome, as both models produce high-dimensional embeddings for semantic search and retrieval.

## Implemented Methods

Each approach is presented in a separate notebook for modular experimentation and comparative evaluation.

### [Method 1: Sentence Transformers + Cosine Similarity](./method_1.ipynb)
Utilizes Hugging Face’s `all-MiniLM-L6-v2` model for local embedding generation.  
Computes cosine similarity using `sklearn.metrics.pairwise` to rank relevant items.  
Outputs the top-ranked fashion products with similarity scores.

### [Method 2: ChromaDB Vector Store](./method_2.ipynb)
Implements **ChromaDB** as a vector storage and retrieval layer.  
Stores precomputed embeddings for scalable and persistent querying.  
Demonstrates efficient semantic retrieval using database-backed vector search.  
Benchmarks retrieval accuracy and similarity relevance.

### [Method 3: RetrievalMind RAG Pipeline (End-to-End AI Recommendation System)](./method_3.ipynb)
Uses my **custom-built and PyPI-published RAG framework: RetrievalMind**:
```bash
pip install RetrievalMind==0.1.3
````

* Implements a complete **end-to-end AI recommendation pipeline**, integrating retrieval, reasoning, and generation.
* Automates embedding generation, vector storage, semantic retrieval, and AI-based response generation.
* Dynamically generates **context-aware AI recommendations** based on user vibe queries.
* Designed for **production-ready deployment**, emphasizing modularity, scalability, and AI-driven personalization.
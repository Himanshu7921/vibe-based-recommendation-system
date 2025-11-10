# Vibe Matcher: Semantic Fashion Recommendation Prototype

Vibe Matcher is a semantic recommendation system prototype that matches fashion products to a user's vibe-based query, such as "energetic urban chic."  
It demonstrates the use of text embeddings and vector search techniques to understand the intent behind natural language queries and retrieve products that best align with the described mood or style.

---

## Project Overview

The project showcases how semantic understanding can enhance traditional search by focusing on *vibe* rather than keyword matching.  
Each product description is embedded as a numerical vector, enabling similarity computation between user input and stored product representations.

---

## Implemented Methods

Each approach is presented in a separate notebook for clarity and modular experimentation.

### [Method 1 — OpenAI Embeddings + Cosine Similarity](./method_1.ipynb)
- Utilizes OpenAI’s `text-embedding-ada-002` model to generate embeddings for product descriptions and user queries.  
- Computes cosine similarity using `sklearn.metrics.pairwise` to determine the most relevant items.  
- Measures runtime with `timeit` for performance evaluation.  
- Outputs the top-ranked products with similarity scores.

---

### [Method 2 — ChromaDB Vector Store](./method_2.ipynb)
- Implements ChromaDB as a vector storage and retrieval system.  
- Stores precomputed embeddings for efficient and scalable querying.  
- Reduces repeated computation and demonstrates database-backed vector search.  
- Evaluates speed and accuracy using the same metrics as Method 1.

---

### [Method 3 — RetrievalMind RAG Pipeline](./method_3.ipynb)
- Uses a custom-built and PyPI-published Retrieval-Augmented Generation (RAG) library:
  ```bash
  pip install RetrievalMind==0.1.3

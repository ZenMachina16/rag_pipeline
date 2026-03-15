import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class ResearchPaperIndexer:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_hnsw=False
    ):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.use_hnsw = use_hnsw

    def build_dense_index(self, chunks):

        texts = [c["text"] for c in chunks]

        print(f"Total chunks to embed: {len(texts)}")

        print("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True   # ⭐ COSINE SIM OPTIMIZATION
        )

        embeddings = np.array(embeddings).astype("float32")

        print("Building FAISS index (Cosine Similarity)...")
        dim = embeddings.shape[1]

        if self.use_hnsw:
            print("Using HNSW Index (Approximate Search)...")
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        else:
            print("Using Flat Inner Product Index...")
            index = faiss.IndexFlatIP(dim)

        index.add(embeddings)

        return index, embeddings

    def build_bm25_index(self, chunks):

        print("Building BM25 index (Sparse Retrieval)...")

        tokenized_corpus = [
            c["text"].lower().split()
            for c in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        return bm25

    def save_all(self, index, bm25, chunks):

        print("Saving FAISS index...")
        faiss.write_index(index, "faiss_index.bin")

        print("Saving BM25 index...")
        with open("bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

        print("Saving chunk store...")
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        print("All indices saved successfully.")


if __name__ == "__main__":

    print("Loading chunks...")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    indexer = ResearchPaperIndexer(
        use_hnsw=False   # ⭐ set True later for scalability
    )

    dense_index, embeddings = indexer.build_dense_index(chunks)

    bm25_index = indexer.build_bm25_index(chunks)

    indexer.save_all(dense_index, bm25_index, chunks)

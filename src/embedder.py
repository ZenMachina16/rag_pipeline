import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class ResearchPaperIndexer:

    def __init__(self):

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_index(self, chunks):

        texts = [c["text"] for c in chunks]

        print("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        print("Building FAISS index...")
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        return index, embeddings

    def save(self, index, chunks):

        faiss.write_index(index, "faiss_index.bin")

        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        print("Index + chunks saved.")


if __name__ == "__main__":

    import pickle

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    indexer = ResearchPaperIndexer()

    index, embeddings = indexer.build_index(chunks)

    indexer.save(index, chunks)

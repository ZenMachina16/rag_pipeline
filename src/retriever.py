import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


class ResearchPaperRetriever:

    def __init__(self):

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading FAISS index...")
        self.index = faiss.read_index("faiss_index.bin")

        print("Loading chunk store...")
        with open("chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query, top_k=5):

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        distances, indices = self.index.search(query_vec, top_k)

        results = []

        for i in indices[0]:
            results.append(self.chunks[i])

        return results


if __name__ == "__main__":

    retriever = ResearchPaperRetriever()

    while True:

        query = input("\nEnter Query: ")

        results = retriever.search(query)

        for i, r in enumerate(results):

            print("\n====================")
            print("Rank:", i+1)
            print("Section:", r["section"])
            print("Text:", r["text"][:400])

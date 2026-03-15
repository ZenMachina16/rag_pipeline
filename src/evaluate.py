import json
import numpy as np

from retriever import ResearchPaperRetriever
import re


TOP_K = 5

from sentence_transformers import SentenceTransformer
model_eval = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def is_relevant(result, gt):

    text = result["text"]
    query = gt["query"]

    q_emb = model_eval.encode(query)
    t_emb = model_eval.encode(text)

    score = np.dot(q_emb, t_emb) / (
        np.linalg.norm(q_emb) * np.linalg.norm(t_emb)
    )

    if score > 0.4:   # threshold tune later
        return True

    return False

def evaluate():

    print("Loading evaluation dataset...")
    with open("data/eval_dataset_v2.json") as f:
        dataset = json.load(f)

    retriever = ResearchPaperRetriever()

    recall_hits = 0
    mrr_total = 0
    hit_rate = 0

    for sample in dataset:

        query = sample["query"]

        print("\n======================================")
        print("Query:", query)

        results = retriever.search(query, top_k=TOP_K)

        found = False

        for rank, r in enumerate(results, start=1):

            if is_relevant(r, sample):

                print(f"✅ Relevant found at rank {rank}")
                recall_hits += 1
                mrr_total += 1 / rank
                hit_rate += 1
                found = True
                break

        if not found:
            print("❌ No relevant chunk found")

    n = len(dataset)

    print("\n\n========== FINAL METRICS ==========")
    print("Queries:", n)
    print("Recall@5:", round(recall_hits / n, 3))
    print("MRR:", round(mrr_total / n, 3))
    print("Hit Rate:", round(hit_rate / n, 3))


if __name__ == "__main__":
    evaluate()

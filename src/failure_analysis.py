import json
import re

from retriever import ResearchPaperRetriever

TOP_K = 5


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text


def is_relevant(result, gt):

    section = normalize(result.get("section", ""))

    # NEW: support both dataset formats

    if "relevance" in gt:

        for s in gt["relevance"].get("sections", []):
            if normalize(s) in section:
                return True

    elif "relevant_section" in gt:

        if normalize(gt["relevant_section"]) in section:
            return True

    return False


def analyze():

    print("Loading evaluation dataset...")
    with open("data/eval_dataset_v2.json") as f:
        dataset = json.load(f)

    retriever = ResearchPaperRetriever()

    failed = 0

    for sample in dataset:

        query = sample["query"]

        results = retriever.search(query, top_k=TOP_K)

        found = False

        for r in results:
            if is_relevant(r, sample):
                found = True
                break

        if found:
            continue

        failed += 1

        print("\n\n========================================")
        print("❌ FAILED QUERY:", query)

        if "relevance" in sample:
            print("Expected Sections:", sample["relevance"].get("sections"))
            print("Expected Keywords:", sample["relevance"].get("keywords"))
        elif "relevant_section" in sample:
            print("Expected Section:", sample["relevant_section"])

        print("\nTop Retrieved Chunks:")

        for i, r in enumerate(results, start=1):

            print("\n------------")
            print("Rank:", i)
            print("Score:", round(r.get("score", 0), 4))
            print("Section:", r["section"])
            print("Text:", r["text"][:300])

    print("\n\nTotal Failed Queries:", failed)
    print("Failure Rate:", round(failed / len(dataset), 3))


if __name__ == "__main__":
    analyze()
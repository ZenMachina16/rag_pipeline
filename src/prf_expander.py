import re
from collections import Counter


STOPWORDS = {
    "the","is","are","of","and","to","in","for","with","that",
    "this","as","an","by","from","be","at","or","it","into",
    "their","which","these","such","can","also","has","have"
}


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text


def extract_expansion_terms(chunks, top_n=5):

    words = []

    for c in chunks:
        text = normalize(c["text"])
        tokens = text.split()

        for t in tokens:
            if t not in STOPWORDS and len(t) > 3:
                words.append(t)

    freq = Counter(words)

    expansion_terms = [w for w, _ in freq.most_common(top_n)]

    return expansion_terms


def expand_query_prf(query, initial_chunks):

    expansion_terms = extract_expansion_terms(initial_chunks)

    expanded_query = query + " " + " ".join(expansion_terms)

    return expanded_query
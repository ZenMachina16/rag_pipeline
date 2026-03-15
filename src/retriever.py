# import faiss
# import pickle
# import numpy as np

# from sentence_transformers import SentenceTransformer


# class ResearchPaperRetriever:

#     def __init__(
#         self,
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         dense_weight=0.7,
#         bm25_weight=0.3
#     ):

#         print("Loading embedding model...")
#         self.model = SentenceTransformer(model_name)

#         print("Loading FAISS index...")
#         self.index = faiss.read_index("faiss_index.bin")

#         print("Loading BM25 index...")
#         with open("bm25.pkl", "rb") as f:
#             self.bm25 = pickle.load(f)

#         print("Loading chunk store...")
#         with open("chunks.pkl", "rb") as f:
#             self.chunks = pickle.load(f)

#         self.dense_weight = dense_weight
#         self.bm25_weight = bm25_weight

#     def dense_search(self, query, top_k=20):

#         query_vec = self.model.encode(
#             [query],
#             normalize_embeddings=True
#         )

#         query_vec = np.array(query_vec).astype("float32")

#         scores, indices = self.index.search(query_vec, top_k)

#         dense_results = {}

#         for score, idx in zip(scores[0], indices[0]):
#             dense_results[idx] = float(score)

#         return dense_results

#     def bm25_search(self, query):

#         tokenized_query = query.lower().split()

#         scores = self.bm25.get_scores(tokenized_query)

#         bm25_results = {
#             i: float(score)
#             for i, score in enumerate(scores)
#         }

#         return bm25_results

#     def hybrid_search(self, query, top_k=5):

#         dense_scores = self.dense_search(query)
#         bm25_scores = self.bm25_search(query)

#         # normalize bm25 scores
#         max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

#         final_scores = {}

#         for idx in range(len(self.chunks)):

#             dense_score = dense_scores.get(idx, 0.0)
#             bm25_score = bm25_scores.get(idx, 0.0) / max_bm25

#             score = (
#                 self.dense_weight * dense_score
#                 + self.bm25_weight * bm25_score
#             )

#             final_scores[idx] = score

#         # sort final scores
#         ranked = sorted(
#             final_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         results = []

#         for idx, score in ranked[:top_k]:
#             chunk = self.chunks[idx].copy()
#             chunk["score"] = score
#             results.append(chunk)

#         return results


# if __name__ == "__main__":

#     retriever = ResearchPaperRetriever()

#     while True:

#         query = input("\nEnter Query: ")

#         results = retriever.hybrid_search(query)

#         for i, r in enumerate(results):

#             print("\n====================")
#             print("Rank:", i + 1)
#             print("Score:", round(r["score"], 4))
#             print("Section:", r["section"])
#             print("Text:", r["text"][:400])
import re
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

NOISE_PATTERNS = [
    re.compile(r'^fig\.\s*\d+', re.IGNORECASE),
    re.compile(r'^figure\s*\d+', re.IGNORECASE),
    re.compile(r'^table\s*\d+', re.IGNORECASE),
    re.compile(r'^results in engineering', re.IGNORECASE),
    re.compile(r'^\[\d+\]'),            # reference entries like [1] Author...
    re.compile(r'^\d{4}$'),             # standalone year
    re.compile(r'^\d+\.\d+$'),          # decimal numbers like 0.845
]

SECTION_WHITELIST = re.compile(
    r'(introduction|background|related work|method|reward|'
    r'experiment|result|discussion|conclusion|planning|'
    r'obstacle|perception|algorithm|workflow|setup|analysis)',
    re.IGNORECASE
)


def is_noise_chunk(chunk: dict) -> bool:
    """
    Returns True if a chunk should be excluded from retrieval.
    Catches figure captions, reference lists, table rows, and
    PDF header/footer artifacts.
    """
    text  = chunk.get("text", "").strip()
    section = chunk.get("section", "").strip()

    # very short chunks carry no useful context
    if len(text.split()) < 20:
        return True

    # chunks whose text is mostly non-ASCII are equation-heavy artifacts
    if len(text) > 0:
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
        if non_ascii_ratio > 0.35:
            return True

    # section label is a bare decimal (e.g. "0.845" from a table cell)
    if re.match(r'^\d+\.\d+$', section):
        return True

    # first line of the chunk matches a known noise pattern
    first_line = text.split("\n")[0].strip()
    if any(p.match(first_line) for p in NOISE_PATTERNS):
        return True

    return False


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ResearchPaperRetriever:

    def __init__(
        self,
        bi_encoder_name  = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_top_k      = 30,    # candidates from dense search
        bm25_top_k       = 30,    # candidates from BM25 search
        rerank_top_k     = 20,    # pool fed to cross-encoder
        final_top_k      = 5,     # results returned to caller
    ):
        print("Loading bi-encoder...")
        self.bi_encoder = SentenceTransformer(bi_encoder_name)

        print("Loading cross-encoder (reranker)...")
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        print("Loading FAISS index...")
        self.index = faiss.read_index("faiss_index.bin")

        print("Loading BM25 index...")
        with open("bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        print("Loading chunk store...")
        with open("chunks.pkl", "rb") as f:
            raw_chunks = pickle.load(f)

        # pre-filter noise chunks once at load time
        self.chunks = raw_chunks
        self.clean_indices = [
            i for i, c in enumerate(raw_chunks)
            if not is_noise_chunk(c)
        ]
        print(f"Chunks after noise filtering: "
              f"{len(self.clean_indices)} / {len(raw_chunks)}")

        self.dense_top_k  = dense_top_k
        self.bm25_top_k   = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k  = final_top_k

    # ------------------------------------------------------------------
    # Stage 1 — Dense retrieval (bi-encoder + FAISS)
    # ------------------------------------------------------------------

    def _dense_search(self, query: str) -> dict[int, float]:
        """
        Returns {chunk_idx: cosine_score} for the top-dense_top_k results.
        Scores are already in [0, 1] due to normalize_embeddings=True.
        """
        vec = self.bi_encoder.encode(
            [query], normalize_embeddings=True
        )
        vec = np.array(vec).astype("float32")

        scores, indices = self.index.search(vec, self.dense_top_k)

        return {
            int(idx): float(score)
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0                          # FAISS returns -1 for empty slots
        }

    # ------------------------------------------------------------------
    # Stage 1 — Sparse retrieval (BM25)
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str) -> dict[int, float]:
        """
        Returns {chunk_idx: raw_bm25_score} for the top-bm25_top_k results.
        """
        tokenized = query.lower().split()
        raw_scores = self.bm25.get_scores(tokenized)

        # only consider clean indices
        filtered = {
            i: float(raw_scores[i])
            for i in self.clean_indices
        }

        # keep top-k by raw score
        top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(top[:self.bm25_top_k])

    # ------------------------------------------------------------------
    # Stage 2 — Hybrid fusion with independent min-max normalisation
    # ------------------------------------------------------------------

    def _hybrid_fusion(
        self,
        dense_scores: dict[int, float],
        bm25_scores:  dict[int, float],
        dense_weight: float = 0.6,
        bm25_weight:  float = 0.4,
    ) -> list[tuple[int, float]]:
        """
        Normalise each signal independently to [0, 1] before weighting.
        This prevents dense's naturally wider range from drowning out BM25.

        Only chunks that appear in either candidate set AND pass the noise
        filter are considered.
        """
        candidate_indices = (
            set(dense_scores.keys()) | set(bm25_scores.keys())
        ) & set(self.clean_indices)

        # --- min-max normalise dense scores ---
        d_vals = [dense_scores[i] for i in candidate_indices if i in dense_scores]
        d_min, d_max = (min(d_vals), max(d_vals)) if d_vals else (0.0, 1.0)
        d_range = d_max - d_min or 1.0

        # --- min-max normalise BM25 scores ---
        b_vals = [bm25_scores[i] for i in candidate_indices if i in bm25_scores]
        b_min, b_max = (min(b_vals), max(b_vals)) if b_vals else (0.0, 1.0)
        b_range = b_max - b_min or 1.0

        fused = {}
        for idx in candidate_indices:
            d_norm = (dense_scores.get(idx, d_min) - d_min) / d_range
            b_norm = (bm25_scores.get(idx,  b_min) - b_min) / b_range
            fused[idx] = dense_weight * d_norm + bm25_weight * b_norm

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:self.rerank_top_k]   # feed only top-N to reranker

    # ------------------------------------------------------------------
    # Stage 3 — Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Score each (query, chunk_text) pair with the cross-encoder.
        Returns re-ranked list of (chunk_idx, cross_encoder_score).
        """
        if not candidates:
            return []

        pairs = [
            (query, self.chunks[idx]["text"])
            for idx, _ in candidates
        ]

        ce_scores = self.cross_encoder.predict(pairs)   # numpy array

        reranked = sorted(
            zip([idx for idx, _ in candidates], ce_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query:          str,
        top_k:          int   = None,
        dense_weight:   float = 0.6,
        bm25_weight:    float = 0.4,
        use_reranker:   bool  = True,
    ) -> list[dict]:
        """
        Full pipeline:
          1. Dense top-30  +  BM25 top-30
          2. Hybrid fusion with independent normalisation  → top-20 pool
          3. Cross-encoder reranking                       → top-k results

        Parameters
        ----------
        query         : natural language question
        top_k         : number of results to return (default: self.final_top_k)
        dense_weight  : weight for normalised dense score  (default 0.6)
        bm25_weight   : weight for normalised BM25 score   (default 0.4)
        use_reranker  : set False to skip cross-encoder (faster, less accurate)
        """
        top_k = top_k or self.final_top_k

        # stage 1
        dense_scores = self._dense_search(query)
        bm25_scores  = self._bm25_search(query)

        # stage 2
        candidates = self._hybrid_fusion(
            dense_scores, bm25_scores,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        # stage 3
        if use_reranker:
            ranked = self._rerank(query, candidates)
        else:
            ranked = candidates     # already sorted by hybrid score

        # assemble results
        results = []
        for idx, score in ranked[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    retriever = ResearchPaperRetriever()

    print("\nRetriever ready. Type 'quit' to exit.")
    print("Options: append  --no-rerank  to skip cross-encoder for speed.\n")

    while True:

        raw = input("Enter Query: ").strip()

        if raw.lower() in ("quit", "exit", "q"):
            break

        use_reranker = True
        query = raw

        if raw.endswith("--no-rerank"):
            use_reranker = False
            query = raw.replace("--no-rerank", "").strip()

        results = retriever.search(query, use_reranker=use_reranker)

        for i, r in enumerate(results):
            print("\n" + "=" * 50)
            print(f"Rank : {i + 1}")
            print(f"Score: {r['score']}")
            print(f"Section: {r['section']}")
            print(f"Text : {r['text'][:400]}")

        print()
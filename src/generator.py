import os
from groq import Groq
from dotenv import load_dotenv

from retriever import ResearchPaperRetriever


# Load ENV
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


class ResearchPaperRAG:

    def __init__(self):

        print("Initializing Groq client...")
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

        print("Loading Retriever...")
        self.retriever = ResearchPaperRetriever()

    # ---------- Context Builder ----------
    def build_context(self, results, max_chunks=4):

        context_parts = []

        for i, r in enumerate(results[:max_chunks], start=1):

            section = r.get("section", "unknown")
            text = r.get("text", "")

            block = f"""
[Chunk {i} | Section: {section}]
{text}
"""
            context_parts.append(block)

        return "\n".join(context_parts)

    # ---------- Prompt Builder ----------
    def build_prompt(self, query, context):

        return f"""
You are a research assistant.

Answer ONLY using the provided context.

Rules:
- Do NOT hallucinate
- If answer not present → say "Not found in paper"
- Always cite chunk number and section
- Be concise but informative
- Merge ideas from multiple chunks

Context:
{context}

Question:
{query}

Answer:
"""

    # ---------- Main Answer Function ----------
    def answer(self, query):

        results = self.retriever.search(query, top_k=10)

        context = self.build_context(results)

        prompt = self.build_prompt(query, context)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )

        return response.choices[0].message.content


if __name__ == "__main__":

    rag = ResearchPaperRAG()

    while True:

        q = input("\nAsk Question: ")

        if q.lower() == "quit":
            break

        ans = rag.answer(q)

        print("\n================ ANSWER ================\n")
        print(ans)
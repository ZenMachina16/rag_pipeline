import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from groq import Groq
from retriever import ResearchPaperRetriever

class ResearchPaperGenerator:

    def __init__(self):

        print("Initializing Groq client...")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        print("Loading Retriever...")
        self.retriever = ResearchPaperRetriever()

    def build_prompt(self, query, chunks):

        context = "\n\n".join(
            [f"[Section: {c['section']}]\n{c['text']}" for c in chunks]
        )

        prompt = f"""


You are a research assistant.

Answer the question ONLY using the provided research paper context.

If answer not found, say "Not enough information".

Question:
{query}

Context:
{context}

Instructions:

* Give structured academic answer
* Mention important findings
* Use technical language
* Be concise
  """

        return prompt


    def answer(self, query):

        chunks = self.retriever.search(query, top_k=5)

        prompt = self.build_prompt(query, chunks)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content


if __name__ == "__main__":

    rag = ResearchPaperGenerator()

    while True:

        q = input("\nAsk Question: ")

        if q == "quit":
            break

        ans = rag.answer(q)

        print("\n================ ANSWER ================\n")
        print(ans)

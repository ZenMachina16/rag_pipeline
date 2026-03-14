import fitz
import re
from typing import List, Dict


class ResearchPaperLoader:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.paper_name = pdf_path.split("/")[-1]

    def load(self) -> List[Dict]:
        doc = fitz.open(self.pdf_path)

        pages_data = []

        for page_num, page in enumerate(doc):
            text = page.get_text()

            text = self.clean_text(text)

            if len(text) < 50:
                continue

            pages_data.append({
                "text": text,
                "page": page_num,
                "paper": self.paper_name
            })

        return pages_data

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # remove references section roughly
        text = re.split(r'References|REFERENCES', text)[0]

        return text.strip()


if __name__ == "__main__":
    loader = ResearchPaperLoader("data/sample.pdf")
    docs = loader.load()

    print("Total Pages Loaded:", len(docs))
    print(docs[0])
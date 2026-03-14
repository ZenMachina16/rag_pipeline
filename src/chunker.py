import re
from typing import List, Dict

# Broader heading pattern — handles unnumbered headings too
HEADING_PATTERNS = [
    re.compile(r'^\d+(\.\d+)*[\.\s]+\w[\w\s]{2,40}$', re.IGNORECASE),   # 1. Introduction
    re.compile(r'^(abstract|introduction|related work|background|'
               r'methodology|methods|experiments?|results?|'
               r'discussion|conclusion|references)\s*$', re.IGNORECASE),  # bare keyword headings
]

NOISE_HEADINGS = {"engineering", "journal", "proceedings", "arxiv", "preprint", "doi"}

def is_heading(line: str) -> bool:
    clean = line.strip()
    if len(clean) < 3 or len(clean) > 80:
        return False
    if any(word in clean.lower() for word in NOISE_HEADINGS):
        return False
    return any(p.match(clean) for p in HEADING_PATTERNS)


class SectionChunker:
    def __init__(self, chunk_size=200, overlap=30, min_chunk_words=30):
        self.chunk_size = chunk_size      # max words per chunk
        self.overlap = overlap            # word overlap between chunks
        self.min_chunk_words = min_chunk_words

    # --- Step 1: Parse full doc into (section, paragraph) pairs ---
    def extract_section_paragraphs(self, pages: List[Dict]) -> List[Dict]:
        """
        Walk every line across all pages.
        Each time a heading is detected, start a new section.
        Accumulate non-empty lines into paragraphs (blank line = paragraph break).
        """
        section_paragraphs = []
        current_section = "unknown"
        current_para_lines = []

        def flush_paragraph():
            text = " ".join(current_para_lines).strip()
            if len(text.split()) >= self.min_chunk_words:
                section_paragraphs.append({
                    "section": current_section,
                    "text": text
                })
            current_para_lines.clear()

        for page in pages:
            for line in page["text"].split("\n"):
                stripped = line.strip()

                if is_heading(stripped):
                    flush_paragraph()
                    current_section = stripped.lower()
                    continue

                if stripped == "":          # blank line = paragraph boundary
                    flush_paragraph()
                else:
                    current_para_lines.append(stripped)

        flush_paragraph()  # don't drop the last paragraph
        return section_paragraphs

    # --- Step 2: Merge small paragraphs, split large ones ---
    def build_chunks(self, section_paragraphs: List[Dict], page_map: Dict) -> List[Dict]:
        """
        Merge consecutive same-section paragraphs until chunk_size is hit,
        then start a new chunk with overlap from the previous one.
        """
        chunks = []
        buffer_words = []
        buffer_section = None

        def flush_buffer(section):
            if len(buffer_words) < self.min_chunk_words:
                return
            chunks.append({
                "text": " ".join(buffer_words),
                "section": section,
                "paper": page_map.get("paper", ""),
            })

        for para in section_paragraphs:
            words = para["text"].split()
            section = para["section"]

            # Section changed → flush immediately, no overlap across sections
            if section != buffer_section:
                flush_buffer(buffer_section)
                buffer_words = []
                buffer_section = section

            # Paragraph fits in current buffer
            if len(buffer_words) + len(words) <= self.chunk_size:
                buffer_words.extend(words)
            else:
                # Flush current buffer, then carry overlap into next chunk
                flush_buffer(buffer_section)
                overlap_words = buffer_words[-self.overlap:] if self.overlap else []
                buffer_words = overlap_words + words

                # If single paragraph still exceeds chunk_size, hard-split it
                while len(buffer_words) > self.chunk_size:
                    chunks.append({
                        "text": " ".join(buffer_words[:self.chunk_size]),
                        "section": section,
                        "paper": page_map.get("paper", ""),
                    })
                    buffer_words = buffer_words[self.chunk_size - self.overlap:]

        flush_buffer(buffer_section)
        return chunks

    def chunk_documents(self, pages: List[Dict]) -> List[Dict]:
        page_map = {"paper": pages[0]["paper"]} if pages else {}
        section_paragraphs = self.extract_section_paragraphs(pages)
        return self.build_chunks(section_paragraphs, page_map)

if __name__ == "__main__":
    import os
    import sys
    # Add project root or src to path so we can import loader if executed directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from loader import ResearchPaperLoader
    
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample.pdf")
    
    print(f"Loading {pdf_path}...")
    loader = ResearchPaperLoader(pdf_path)
    pages = loader.load()
    
    print("Chunking documents...")
    chunker = SectionChunker()
    chunks = chunker.chunk_documents(pages)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print("="*40)
    for i, chunk in enumerate(chunks):
        print("\n==============================")
        print(f"Chunk {i}")
        print("Section:", chunk["section"])
        print("Paper:", chunk["paper"])
        print("Text:", chunk["text"])


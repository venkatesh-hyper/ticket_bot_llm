import os
import re
from PyPDF2 import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from app.vector_store import VectorStore


class PDFIngestor:
    """Extracts, cleans, chunks, embeds, and stores PDF text into ChromaDB."""

    def __init__(self, chunk_size=800, chunk_overlap=100, model_name="all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = SentenceTransformer(model_name)
        self.store = VectorStore("chroma_storage")

    def _clean_text(self, text):
        """Normalize whitespace and remove unwanted characters."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9.,!?;:\-()&%$#@'\" ]+", " ", text)
        return text.strip()

    def _chunk_text(self, text):
        """Split long text into overlapping semantic chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if len(chunk) > 50:  # avoid tiny fragments
                chunks.append(chunk)
        return chunks

    def process_pdf(self, pdf_path):
        """Main entry — read, embed, and store PDF chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        print(f"📖 Reading and embedding PDF: {pdf_path}")
        all_chunks, all_metadatas = [], []

        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for i, page in enumerate(tqdm(reader.pages, desc="Extracting pages")):
                try:
                    raw = page.extract_text() or ""
                    cleaned = self._clean_text(raw)
                    for chunk in self._chunk_text(cleaned):
                        all_chunks.append(chunk)
                        all_metadatas.append({"page": i + 1})
                except Exception as e:
                    print(f"Page {i + 1} extraction error: {e}")

        if not all_chunks:
            print("No chunks extracted. Check PDF readability.")
            return

        print(f"Embedding {len(all_chunks)} text chunks...")
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True)

        print("Storing embeddings into ChromaDB...")
        self.store.add_chunks(all_chunks, embeddings, all_metadatas)
        print("Ingestion complete! Your data is now ready for retrieval.")


# ✅ FIXED: removed self-import that caused silent exit
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    try:
        ingestor = PDFIngestor()
        ingestor.process_pdf(pdf_path)
    except Exception as e:
        print(f"Ingestion failed: {e}")

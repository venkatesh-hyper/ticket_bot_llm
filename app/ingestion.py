import PyPDF2
import os
import re

class PDFIngestor:
    """Stream-reads a PDF and yields clean text per page."""

    def extract_text(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        print(f"Reading {pdf_path} ...")
        reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
        for i, page in enumerate(reader.pages):
            try:
                raw = page.extract_text() or ""
                clean = re.sub(r"\s+", " ", raw).strip()
                if clean:
                    yield {"page": i + 1, "text": clean}
            except Exception as e:
                print(f"⚠️ Page {i+1} extraction error: {e}")

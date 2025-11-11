from app.ingestion import PDFIngestor
from app.embedder import Embedder
from app.vector_store import VectorStore
from tqdm import tqdm

pdf_path = "tickets.pdf"

ingestor = PDFIngestor()
embedder = Embedder()
store = VectorStore()

batch_texts, batch_meta = [], []

for page in tqdm(ingestor.extract_text(pdf_path), desc="Processing pages"):
    chunks = embedder.chunk_text(page["text"])
    batch_texts.extend(chunks)
    batch_meta.extend([{"page": page["page"]}] * len(chunks))

# embed in manageable batches
batch_size = 200
for i in range(0, len(batch_texts), batch_size):
    sub_texts = batch_texts[i:i+batch_size]
    embs = embedder.embed_batch(sub_texts)
    store.add_chunks(sub_texts, embs, batch_meta[i:i+batch_size])

print("Indexing complete.")


import json
import math
from pathlib import Path

import ollama
from pypdf import PdfReader

# Use the models from the previous RAG tutorial.
# Change these if "ollama list" shows different names.
EMBED_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
CHAT_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

BASE_DIR = Path(__file__).resolve().parent
MANUAL_DIR = BASE_DIR / "manuals"
INDEX_FILE = BASE_DIR / "index.json"


def split_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_index():
    documents = []
    pdf_files = sorted(MANUAL_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the manuals folder."
        )

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk in split_text(text):
                documents.append({
                    "file": pdf_path.name,
                    "page": page_number,
                    "text": chunk
                })

    if not documents:
        raise ValueError(
            "No readable text found in the PDF files."
        )

    print(f"Creating embeddings for {len(documents)} chunks...")

    for start in range(0, len(documents), 16):
        batch = documents[start:start + 16]
        result = ollama.embed(
            model=EMBED_MODEL,
            input=[doc["text"] for doc in batch]
        )
        for doc, vector in zip(batch, result["embeddings"]):
            doc["embedding"] = vector
        print(f"Processed {start + len(batch)}/{len(documents)}")

    INDEX_FILE.write_text(
        json.dumps({
            "embedding_model": EMBED_MODEL,
            "documents": documents
        }),
        encoding="utf-8"
    )
    return documents


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


def search_documents(question, documents, top_k=4):
    query_vector = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )["embeddings"][0]

    ranked = sorted(
        documents,
        key=lambda doc: cosine_similarity(
            query_vector, doc["embedding"]
        ),
        reverse=True
    )
    return ranked[:top_k]


def answer_question(question, documents):
    matches = search_documents(question, documents)

    context_parts = []
    for i, doc in enumerate(matches, start=1):
        context_parts.append(
            f"[Source {i}: {doc['file']}, page {doc['page']}]\n"
            f"{doc['text']}"
        )

    context = "\n\n".join(context_parts)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical manual assistant. "
                    "Answer only using the provided manual excerpts. "
                    "If the information is missing, say you cannot "
                    "find it in the manual. Cite the source number "
                    "and page number. Do not invent specifications."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Manual excerpts:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    print("\nANSWER:")
    print(response["message"]["content"])

    print("\nRETRIEVED SOURCES:")
    for i, doc in enumerate(matches, start=1):
        print(f"[{i}] {doc['file']} - Page {doc['page']}")


def main():
    MANUAL_DIR.mkdir(exist_ok=True)

    if INDEX_FILE.exists():
        saved = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if saved.get("embedding_model") != EMBED_MODEL:
            raise ValueError(
                "Embedding model changed. Delete index.json and rerun."
            )
        documents = saved["documents"]
        print("Loaded existing document index.")
    else:
        documents = build_index()

    while True:
        question = input("\nAsk a question (or type exit): ").strip()

        if question.lower() in ("exit", "quit"):
            break

        if question:
            answer_question(question, documents)


if __name__ == "__main__":
    main()

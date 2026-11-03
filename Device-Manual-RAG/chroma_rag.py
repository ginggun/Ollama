
from pathlib import Path
import sys

import chromadb
import ollama
from pypdf import PdfReader

# Check these model names with: ollama list
EMBED_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
CHAT_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "Manuals"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "device_manuals"

# Local persistent ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}}
    )


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


def ingest_pdfs():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in manuals folder.")
        return

    # Rebuild our dedicated collection.
    # This removes its previous records.

    # Delete the collection only if it already exists
    if COLLECTION_NAME in [
        col.name for col in client.list_collections()
    ]:
        client.delete_collection(name=COLLECTION_NAME)

    collection = get_collection()


    total = 0

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks = split_text(text)

            if not chunks:
                continue

            # Generate embeddings using Ollama
            response = ollama.embed(
                model=EMBED_MODEL,
                input=chunks
            )

            embeddings = response["embeddings"]

            ids = [
                f"{pdf_path.name}-p{page_num}-c{i}"
                for i in range(len(chunks))
            ]

            metadatas = [
                {
                    "source": pdf_path.name,
                    "page": page_num,
                    "embedding_model": EMBED_MODEL
                }
                for _ in chunks
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            total += len(chunks)

        print(f"Finished: {pdf_path.name}")

    print(f"\nSaved {total} chunks to ChromaDB!")


def ask_question(question):
    collection = get_collection()

    count = collection.count()

    if count == 0:
        print("Database is empty. Run ingest first.")
        return

    # Convert the question into an embedding
    query_embedding = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )["embeddings"][0]

    # Search the ChromaDB database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(4, count),
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []

    for i, (doc, metadata) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0]
        ),
        start=1
    ):
        context_parts.append(
            f"[Source {i}: {metadata['source']}, "
            f"page {metadata['page']}]\n{doc}"
        )

    context = "\n\n".join(context_parts)

    # Ollama uses retrieved context to answer
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a device manual assistant. "
                    "Use only the provided manual excerpts. "
                    "If the answer is not in the excerpts, "
                    "say that the manual does not provide it. "
                    "Cite the source number and page. "
                    "Never invent specifications."
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

    print("\n========== AI ANSWER ==========")
    print(response["message"]["content"])

    print("\n========== SOURCES ==========")

    for metadata in results["metadatas"][0]:
        print(
            f"{metadata['source']} - "
            f"Page {metadata['page']}"
        )


def main():
    PDF_DIR.mkdir(exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chroma_rag.py ingest")
        print("  python chroma_rag.py ask")
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        ingest_pdfs()

    elif command == "ask":
        print("ChromaDB Manual Assistant")
        print("Type 'exit' to quit.")

        while True:
            question = input("\nQuestion: ").strip()

            if question.lower() in ("exit", "quit"):
                break

            if question:
                ask_question(question)

    else:
        print("Unknown command. Use ingest or ask.")


if __name__ == "__main__":
    main()

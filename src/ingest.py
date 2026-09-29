from pathlib import Path

import numpy as np

from foundry_local_sdk import (
    Configuration,
    EmbeddingsSession,
    FoundryLocalManager,
    Request,
    TensorItem,
    TextItem,
)

from database import (
    get_connection,
    initialize_database,
    insert_document,
)


DOCUMENTS_DIR = Path("data/documents")
EMBEDDING_MODEL = "qwen3-embedding-0.6b"


def chunk_text(text, max_chars=700):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def load_documents():
    documents = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8-sig")

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            documents.append(
                {
                    "source": path.name,
                    "chunk_index": index,
                    "content": chunk,
                }
            )

    return documents


def generate_embeddings(model, texts):
    embeddings = []

    with EmbeddingsSession(model) as session:
        with Request() as request:

            for text in texts:
                request.add_item(TextItem(text))

            with session.process_request(request) as response:
                for item in response:

                    if isinstance(item, TensorItem):
                        vector = np.frombuffer(
                            item.data,
                            dtype=np.float32
                        ).copy()

                        embeddings.append(vector)

    return embeddings


def clear_sources(documents):
    sources = sorted(
        set(document["source"] for document in documents)
    )

    with get_connection() as conn:
        for source in sources:
            conn.execute(
                "DELETE FROM documents WHERE source = ?",
                (source,),
            )

        conn.commit()


def main():
    print("1 - Veritabani hazirlaniyor")
    initialize_database()

    print("2 - Dokumanlar okunuyor")
    documents = load_documents()

    if not documents:
        print("HATA: data/documents klasorunde .txt dosyasi bulunamadi.")
        return

    print("Toplam chunk:", len(documents))

    print("3 - Foundry Local baslatiliyor")

    FoundryLocalManager.initialize(
        Configuration(app_name="LocalRAGIngestion")
    )

    manager = FoundryLocalManager.instance

    print("4 - Embedding modeli aliniyor")
    model = manager.catalog.get_model(EMBEDDING_MODEL)

    print("5 - Model kontrol ediliyor")
    model.download(
        lambda pct: print(
            f"\rIndiriliyor: {pct:.1f}%",
            end="",
            flush=True,
        )
    )

    print("\n6 - Model yukleniyor")
    model.load()

    print("7 - Embeddingler uretiliyor")

    texts = [
        document["content"]
        for document in documents
    ]

    embeddings = generate_embeddings(
        model,
        texts,
    )

    if len(embeddings) != len(documents):
        model.unload()

        raise RuntimeError(
            "Embedding sayisi ile chunk sayisi eslesmiyor."
        )

    print("8 - SQLite kayitlari yenileniyor")

    clear_sources(documents)

    for document, embedding in zip(
        documents,
        embeddings,
    ):
        insert_document(
            source=document["source"],
            chunk_index=document["chunk_index"],
            content=document["content"],
            embedding=embedding,
        )

        print(
            f"Kaydedildi: "
            f"{document['source']} "
            f"- chunk {document['chunk_index']}"
        )

    model.unload()

    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

    print()
    print("9 - Ingestion tamamlandi")
    print("Veritabanindaki toplam chunk:", count)


if __name__ == "__main__":
    main()

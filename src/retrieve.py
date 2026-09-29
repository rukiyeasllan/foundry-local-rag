import numpy as np

from foundry_local_sdk import (
    Configuration,
    EmbeddingsSession,
    FoundryLocalManager,
    Request,
    TensorItem,
    TextItem,
)

from database import get_all_documents


EMBEDDING_MODEL = "qwen3-embedding-0.6b"


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def generate_query_embedding(model, query):
    with EmbeddingsSession(model) as session:
        with Request().add_item(TextItem(query)) as request:
            with session.process_request(request) as response:

                for item in response:
                    if isinstance(item, TensorItem):
                        return np.frombuffer(
                            item.data,
                            dtype=np.float32
                        ).copy()

    raise RuntimeError("Sorgu embeddingi uretilemedi.")


def retrieve(query, top_k=2):
    documents = get_all_documents()

    if not documents:
        raise RuntimeError(
            "Veritabaninda dokuman bulunamadi. "
            "Once ingest.py calistirin."
        )

    print("1 - Foundry Local baslatiliyor")

    FoundryLocalManager.initialize(
        Configuration(app_name="LocalRAGRetrieval")
    )

    manager = FoundryLocalManager.instance

    print("2 - Embedding modeli aliniyor")

    model = manager.catalog.get_model(
        EMBEDDING_MODEL
    )

    print("3 - Model kontrol ediliyor")

    model.download(
        lambda pct: print(
            f"\rIndiriliyor: {pct:.1f}%",
            end="",
            flush=True
        )
    )

    print("\n4 - Model yukleniyor")
    model.load()

    print("5 - Sorgu embeddingi uretiliyor")

    query_embedding = generate_query_embedding(
        model,
        query
    )

    results = []

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append(
            {
                "score": score,
                "source": document["source"],
                "chunk_index": document["chunk_index"],
                "content": document["content"],
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    model.unload()

    return results[:top_k]


if __name__ == "__main__":

    query = "Foundry Local ne ise yarar?"

    print()
    print("Soru:", query)
    print()

    results = retrieve(
        query,
        top_k=2
    )

    print()
    print("En ilgili sonuclar:")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"SONUC {rank}")
        print(
            "Benzerlik:",
            round(result["score"], 4)
        )
        print(
            "Kaynak:",
            result["source"]
        )
        print(
            "Chunk:",
            result["chunk_index"]
        )
        print()
        print(result["content"])
        print()
        print("-" * 60)

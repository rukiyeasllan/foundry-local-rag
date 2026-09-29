import numpy as np

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
    EmbeddingsSession,
    Request,
    TensorItem,
    TextItem,
)

print("1 - Foundry Local baslatiliyor")

FoundryLocalManager.initialize(
    Configuration(app_name="LocalRAGEmbeddings")
)

manager = FoundryLocalManager.instance

print("2 - Embedding modeli aliniyor")
model = manager.catalog.get_model("qwen3-embedding-0.6b")

print("3 - Model indiriliyor")
model.download(
    lambda pct: print(
        f"\rIndiriliyor: {pct:.1f}%",
        end="",
        flush=True
    )
)

print("\n4 - Model yukleniyor")
model.load()

print("5 - Embedding uretiliyor")

texts = [
    "Yapay zeka bilgisayar sistemlerinin insan benzeri gorevler yapmasini saglar.",
    "Makine ogrenmesi yapay zekanin bir alt alanidir.",
    "Bugun hava oldukca guzel."
]

with EmbeddingsSession(model) as session:
    with Request() as req:

        for text in texts:
            req.add_item(TextItem(text))

        with session.process_request(req) as response:

            index = 0

            for item in response:

                if isinstance(item, TensorItem):

                    embedding = np.frombuffer(
                        item.data,
                        dtype=np.float32
                    )

                    print(f"\nMetin {index + 1}: {texts[index]}")
                    print("Embedding boyutu:", len(embedding))
                    print("Ilk 5 deger:", embedding[:5])

                    index += 1

model.unload()

print("\n6 - Embedding testi tamamlandi")

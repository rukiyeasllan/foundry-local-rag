import numpy as np

from foundry_local_sdk import (
    ChatSession,
    Configuration,
    EmbeddingsSession,
    FoundryLocalManager,
    MessageItem,
    Request,
    RequestOptions,
    SearchOptions,
    TensorItem,
    TextItem,
)

from src.database import get_all_documents


EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-4-mini"

TOP_K = 2
MIN_SIMILARITY = 0.35

FALLBACK = "Bu bilgi mevcut belgelerde bulunamadi."


class LocalRAG:
    def __init__(self):
        print("Local RAG sistemi baslatiliyor...")

        FoundryLocalManager.initialize(
            Configuration(app_name="LocalRAGAssistant")
        )

        self.manager = FoundryLocalManager.instance

        self.documents = get_all_documents()

        if not self.documents:
            raise RuntimeError(
                "Veritabaninda dokuman yok. "
                "Once python src\\ingest.py calistirin."
            )

        print(
            f"Veritabaninda {len(self.documents)} chunk bulundu."
        )

        self.embedding_model = self.manager.catalog.get_model(
            EMBEDDING_MODEL
        )

        self.chat_model = self.manager.catalog.get_model(
            CHAT_MODEL
        )

        print("Embedding modeli kontrol ediliyor...")

        self.embedding_model.download(
            lambda pct: print(
                f"\rEmbedding modeli: {pct:.1f}%",
                end="",
                flush=True
            )
        )

        print()

        print("Chat modeli kontrol ediliyor...")

        self.chat_model.download(
            lambda pct: print(
                f"\rChat modeli: {pct:.1f}%",
                end="",
                flush=True
            )
        )

        print()
        print("Embedding modeli bellege yukleniyor...")
        self.embedding_model.load()

        print("Chat modeli bellege yukleniyor...")
        self.chat_model.load()
        print("Sistem hazir.")


    @staticmethod
    def cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )


    def embed_query(self, query):
        try:
            with EmbeddingsSession(
                self.embedding_model
            ) as session:

                with Request().add_item(
                    TextItem(query)
                ) as request:

                    with session.process_request(
                        request
                    ) as response:

                        for item in response:

                            if isinstance(
                                item,
                                TensorItem
                            ):
                                return np.frombuffer(
                                    item.data,
                                    dtype=np.float32
                                ).copy()

        finally:
            pass

        raise RuntimeError(
            "Sorgu embeddingi uretilemedi."
        )


    def retrieve(self, query):
        query_embedding = self.embed_query(query)

        results = []

        for document in self.documents:

            score = self.cosine_similarity(
                query_embedding,
                document["embedding"]
            )

            results.append(
                {
                    "score": score,
                    "source": document["source"],
                    "chunk_index": document[
                        "chunk_index"
                    ],
                    "content": document["content"],
                }
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return results[:TOP_K]


    @staticmethod
    def build_context(results):
        parts = []

        for index, result in enumerate(
            results,
            start=1
        ):
            parts.append(
                f"[KAYNAK {index}]\n"
                f"Dosya: {result['source']}\n"
                f"Chunk: {result['chunk_index']}\n"
                f"Icerik:\n{result['content']}"
            )

        return "\n\n".join(parts)


    def generate_answer(
        self,
        question,
        results
    ):
        context = self.build_context(results)

        system_prompt = (
            "Sen yerel ve belge tabanli bir "
            "RAG soru-cevap asistanisin. "
            "Yalnizca verilen KAYNAKLARDAKI "
            "bilgileri kullan. "
            "Kaynaklarda olmayan bilgi ekleme. "
            "Tahmin yapma. "
            "Cevabi Turkce, acik ve en fazla "
            "2 kisa cumle halinde ver. 50 kelimeyi gecme. "
            "Gereksiz tekrar yapma."
        )

        user_prompt = (
            f"KAYNAKLAR:\n{context}\n\n"
            f"SORU:\n{question}\n\n"
            "Soruyu dogrudan cevapla."
        )

        answer_parts = []

        try:
            with ChatSession(
                self.chat_model
            ) as session:

                session.set_options(
                    RequestOptions(
                        search=SearchOptions(
                            temperature=0.0,
                            max_output_tokens=80
                        )
                    )
                )

                with Request() as request:

                    request.add_item(
                        MessageItem.system(
                            system_prompt
                        )
                    )

                    request.add_item(
                        MessageItem.user(
                            user_prompt
                        )
                    )

                    with session.process_request(
                        request
                    ) as response:

                        for item in response:

                            if isinstance(
                                item,
                                TextItem
                            ):
                                answer_parts.append(
                                    item.text
                                )

                            elif isinstance(
                                item,
                                MessageItem
                            ):
                                for part in item.parts:

                                    if isinstance(
                                        part,
                                        TextItem
                                    ):
                                        answer_parts.append(
                                            part.text
                                        )

        finally:
            pass

        answer = "".join(
            answer_parts
        ).strip()

        if not answer:
            return "Model cevap uretemedi."

        if (
            FALLBACK in answer
            and answer != FALLBACK
        ):
            answer = answer.replace(
                FALLBACK,
                ""
            ).strip()

        return answer


    def ask(self, question):
        results = self.retrieve(question)

        if not results:
            return FALLBACK, []

        best_score = results[0]["score"]

        if best_score < MIN_SIMILARITY:
            return FALLBACK, []

        relevant_results = [
            result
            for result in results
            if result["score"]
            >= MIN_SIMILARITY
        ]

        answer = self.generate_answer(
            question,
            relevant_results
        )

        return answer, relevant_results


def main():
    print()
    print("=" * 60)
    print("LOCAL RAG AI ASSISTANT")
    print("Microsoft Foundry Local")
    print("=" * 60)
    print()

    rag = LocalRAG()

    print()
    print(
        "Sorunuzu yazabilirsiniz."
    )
    print(
        "Cikmak icin 'cikis' yazin."
    )

    while True:
        print()

        question = input("Siz > ").strip()

        if not question:
            continue

        if question.lower() in {
            "cikis",
            "exit",
            "quit"
        }:
            print("Program kapatildi.")
            break

        try:
            answer, sources = rag.ask(
                question
            )

            print()
            print("Asistan >")
            print(answer)

            if sources:
                print()
                print("Kaynaklar:")

                for index, source in enumerate(
                    sources,
                    start=1
                ):
                    print(
                        f"[{index}] "
                        f"{source['source']} "
                        f"(chunk "
                        f"{source['chunk_index']}, "
                        f"skor="
                        f"{source['score']:.4f})"
                    )

        except Exception as error:
            print()
            print(
                "HATA:",
                error
            )


if __name__ == "__main__":
    main()










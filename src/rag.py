from foundry_local_sdk import (
    ChatSession,
    FoundryLocalManager,
    MessageItem,
    Request,
    RequestOptions,
    SearchOptions,
    TextItem,
)

from retrieve import retrieve


CHAT_MODEL = "phi-4-mini"
MIN_SIMILARITY = 0.35


def build_context(results):
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"[KAYNAK {index}]\n"
            f"Dosya: {result['source']}\n"
            f"Icerik: {result['content']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, results):
    manager = FoundryLocalManager.instance

    print("6 - Chat modeli aliniyor:", CHAT_MODEL)
    model = manager.catalog.get_model(CHAT_MODEL)

    print("7 - Chat modeli kontrol ediliyor")
    model.download(
        lambda pct: print(
            f"\rIndiriliyor: {pct:.1f}%",
            end="",
            flush=True
        )
    )

    print("\n8 - Chat modeli yukleniyor")
    model.load()

    context = build_context(results)

    system_prompt = (
        "Sen belge tabanli bir soru-cevap asistanisin. "
        "Yalnizca verilen kaynak metnindeki bilgileri kullan. "
        "Kaynakta olmayan bilgi ekleme veya tahmin etme. "
        "Sorunun cevabi kaynakta yoksa tam olarak "
        "'Bu bilgi mevcut belgelerde bulunamadi.' yaz. "
        "Cevabin en fazla 3 cumle olsun. "
        "Tekrar yapma. "
        "Turkce cevap ver."
    )

    user_prompt = (
        f"KAYNAKLAR:\n{context}\n\n"
        f"SORU:\n{question}\n\n"
        "Soruyu dogrudan cevapla. "
        "Aciklama gerekmiyorsa ek bilgi verme."
    )

    answer_parts = []

    with ChatSession(model) as session:
        session.set_options(
            RequestOptions(
                search=SearchOptions(
                    temperature=0.0,
                    max_output_tokens=120
                )
            )
        )

        with Request() as request:
            request.add_item(
                MessageItem.system(system_prompt)
            )

            request.add_item(
                MessageItem.user(user_prompt)
            )

            with session.process_request(request) as response:
                for item in response:

                    if isinstance(item, TextItem):
                        answer_parts.append(item.text)

                    elif isinstance(item, MessageItem):
                        for part in item.parts:
                            if isinstance(part, TextItem):
                                answer_parts.append(part.text)

    model.unload()

    answer = "".join(answer_parts).strip()

    if not answer:
        return "Model cevap uretemedi."

    fallback = "Bu bilgi mevcut belgelerde bulunamadi."

    # Model gecerli bir cevap verdikten sonra fallback cumlesini
    # gereksiz yere eklerse temizle.
    if fallback in answer and answer.strip() != fallback:
        answer = answer.replace(fallback, "").strip()

    return answer


def answer_question(question):
    print()
    print("SORU:", question)
    print()

    results = retrieve(
        question,
        top_k=2
    )

    if not results:
        return (
            "Bu bilgi mevcut belgelerde bulunamadi.",
            []
        )

    best_score = results[0]["score"]

    print()
    print(
        "En yuksek benzerlik:",
        round(best_score, 4)
    )

    if best_score < MIN_SIMILARITY:
        return (
            "Bu bilgi mevcut belgelerde bulunamadi.",
            []
        )

    # Generation icin yalnizca gercekten ilgili chunklari kullan.
    relevant_results = [
        result
        for result in results
        if result["score"] >= MIN_SIMILARITY
    ]

    answer = generate_answer(
        question,
        relevant_results
    )

    return answer, relevant_results


if __name__ == "__main__":

    question = "Turkiye Cumhuriyetinin baskenti neresidir?"

    answer, sources = answer_question(question)

    print()
    print("=" * 60)
    print("RAG CEVABI")
    print("=" * 60)
    print()
    print(answer)

    print()
    print("=" * 60)
    print("KAYNAKLAR")
    print("=" * 60)

    for index, source in enumerate(
        sources,
        start=1
    ):
        print(
            f"[{index}] "
            f"{source['source']} "
            f"(chunk {source['chunk_index']}, "
            f"benzerlik={source['score']:.4f})"
        )


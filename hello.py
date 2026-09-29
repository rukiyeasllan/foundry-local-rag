from foundry_local_sdk import (
    ChatSession,
    Configuration,
    FoundryLocalManager,
    MessageItem,
    Request,
    RequestOptions,
    SearchOptions,
    TextItem,
)

print("1 - Foundry Local baslatiliyor")

FoundryLocalManager.initialize(
    Configuration(app_name="LocalRAG")
)

manager = FoundryLocalManager.instance

print("2 - Model aliniyor")

model = manager.catalog.get_model("qwen2.5-0.5b")

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

print("5 - Model yuklendi")
print("6 - Soru gonderiliyor")

with ChatSession(model) as session:

    session.set_options(
        RequestOptions(
            search=SearchOptions(
                temperature=0.0,
                max_output_tokens=128
            )
        )
    )

    with Request().add_item(
        MessageItem.user(
            "Merhaba. Kendini bir cumleyle tanit."
        )
    ) as req:

        with session.process_request(req) as response:

            print("7 - Model cevabi:")

            for item in response:

                if isinstance(item, TextItem):
                    print(item.text)

                elif isinstance(item, MessageItem):

                    for part in item.parts:

                        if isinstance(part, TextItem):
                            print(part.text)

            print("Finish reason:", response.finish_reason)

model.unload()

print("8 - Test tamamlandi")
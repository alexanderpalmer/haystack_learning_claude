from haystack.dataclasses import ChatMessage
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

print("=== Embedding-Test ===")

embedder = OllamaTextEmbedder(model="qwen3-embedding:0.6b")
embed_result = embedder.run(
    text="Haystack ist ein Framework für Retrieval-Augmented Generation."
)

embedding = embed_result["embedding"]
print(f"Embedding-Länge: {len(embedding)}")
print(f"Erste 5 Werte: {embedding[:5]}")

print("\n=== Chat-Test ===")

llm = OllamaChatGenerator(
    model="llama3.2:3b",
    generation_kwargs={
        "temperature": 0
    }
)

messages = [
    ChatMessage.from_system(
        "Du bist ein präziser Assistent. "
        "Antworte kurz, sachlich und auf Deutsch. "
        "Wenn ein Begriff mehrdeutig ist, erkläre nur die Bedeutung im KI-Kontext."
    ),
    ChatMessage.from_user(
        "Erkläre in einem Satz, was Retrieval-Augmented Generation (RAG) im KI-Kontext ist."
    )
]

chat_result = llm.run(messages=messages)
reply = chat_result["replies"][0]

print(reply.text)
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

INDEX_PATH = "storage/buch_index"

document_store = FAISSDocumentStore(
    index_path=INDEX_PATH,
    embedding_dim=1024,
)

template = [
    ChatMessage.from_system(
        "Du bist ein technischer Assistent für das Buch "
        "'Python für KI- und Daten-Projekte' von Johannes Schildgen.\n\n"
        "Beantworte die Frage ausführlich und präzise anhand der bereitgestellten Buchauszüge.\n"
        "Regeln:\n"
        "- Beziehe dich nur auf Informationen aus den Buchauszügen.\n"
        "- Wenn der Kontext Code-Beispiele enthält, gib sie vollständig wieder.\n"
        "- Wenn mehrere Abschnitte relevant sind, fasse sie zusammen.\n"
        "- Gib am Ende die Seitenzahlen der verwendeten Quellen an (z.B. 'Quellen: Seite 42, 43').\n"
        "- Wenn die Antwort nicht im Kontext enthalten ist, antworte: "
        "'Dazu enthält das Buch in den gefundenen Abschnitten keine Information.'"
    ),
    ChatMessage.from_user(
        "Frage: {{ question }}\n\n"
        "Buchauszüge:\n"
        "{% for doc in documents %}"
        "--- Seite {{ doc.meta.page_number }} ---\n"
        "{{ doc.content }}\n\n"
        "{% endfor %}"
        "Beantworte die Frage basierend auf den obigen Auszügen."
    ),
]

query_pipeline = Pipeline()

query_pipeline.add_component(
    "text_embedder",
    OllamaTextEmbedder(model="qwen3-embedding:0.6b"),
)

query_pipeline.add_component(
    "retriever",
    FAISSEmbeddingRetriever(document_store=document_store),
)

query_pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(
        template=template,
        required_variables={"question", "documents"},
    ),
)

query_pipeline.add_component(
    "llm",
    OllamaChatGenerator(
        model="gemma3:4b",
        timeout=600,
        generation_kwargs={"temperature": 0.1},
    ),
)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "llm.messages")

question = input("Frage eingeben: ").strip()

print("\n=== Frage ===")
print(question)

result = query_pipeline.run(
    data={
        "text_embedder": {"text": question},
        "retriever": {"top_k": 8},
        "prompt_builder": {"question": question},
    },
    include_outputs_from={"retriever"},
)

print("\n=== Antwort ===")
print(result["llm"]["replies"][0].text)

print("\n=== Gefundene Chunks ===")
for i, doc in enumerate(result["retriever"]["documents"], start=1):
    page = doc.meta.get("page_number", "?")
    score = getattr(doc, "score", None)
    score_str = f"{score:.4f}" if score is not None else "n/a"
    print(f"\nChunk {i} | Seite: {page} | Score: {score_str}")
    print(doc.content[:300])

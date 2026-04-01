from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)
from haystack_integrations.components.generators.ollama import OllamaChatGenerator


# --------------------------------------------------
# 1) Document Store
# --------------------------------------------------
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")


# --------------------------------------------------
# 2) Indexing-Pipeline
# --------------------------------------------------
indexing_pipeline = Pipeline()

indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component(
    "splitter",
    DocumentSplitter(
        split_by="word",
        split_length=200,
        split_overlap=30,
    ),
)
indexing_pipeline.add_component(
    "embedder",
    OllamaDocumentEmbedder(model="qwen3-embedding:0.6b"),
)
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    ),
)

indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")


# --------------------------------------------------
# 3) Query-Pipeline
# --------------------------------------------------
# --------------------------------------------------
# 3) Query-Pipeline
# --------------------------------------------------
template = [
    ChatMessage.from_system(
        "Du beantwortest Fragen ausschließlich anhand des bereitgestellten Kontexts. "
        "Verwende keine eigenen Annahmen und ergänze nichts aus Weltwissen. "
        "Antworte nur das, was im Kontext direkt gestützt ist. "
        "Wenn die Information nicht klar im Kontext steht, sage exakt: "
        "'Dazu finde ich im Dokument keine ausreichende Information.' "
        "Antworte auf Deutsch, kurz und präzise. "
        "Nenne am Ende die Seitenzahlen in der Form: Quelle: Seite X."
    ),
    ChatMessage.from_user(
        "Frage: {{ question }}\n\n"
        "Kontext:\n"
        "{% for doc in documents %}"
        "-----\n"
        "Seite {{ doc.meta.page_number }}:\n"
        "{{ doc.content }}\n"
        "{% endfor %}\n\n"
        "Beantworte die Frage nur mit Hilfe dieses Kontexts."
    ),
]

query_pipeline = Pipeline()

query_pipeline.add_component(
    "bm25_retriever",
    InMemoryBM25Retriever(document_store=document_store),
)

query_pipeline.add_component(
    "text_embedder",
    OllamaTextEmbedder(model="qwen3-embedding:0.6b"),
)

query_pipeline.add_component(
    "embedding_retriever",
    InMemoryEmbeddingRetriever(document_store=document_store),
)

query_pipeline.add_component(
    "joiner",
    DocumentJoiner(join_mode="reciprocal_rank_fusion"),
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
        model="llama3.2:3b",
        timeout=300,
        generation_kwargs={"temperature": 0},
    ),
)

query_pipeline.connect("bm25_retriever", "joiner")
query_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
query_pipeline.connect("embedding_retriever", "joiner")
query_pipeline.connect("joiner.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "llm.messages")


# --------------------------------------------------
# 4) PDF indexieren
# --------------------------------------------------
print("=== Indexierung startet ===")
indexing_result = indexing_pipeline.run(
    {
        "converter": {
            "sources": ["data/buch.pdf"]
        }
    }
)
print(f"Geschriebene Dokumente: {indexing_result['writer']['documents_written']}")


# --------------------------------------------------
# 5) Testfrage stellen
# --------------------------------------------------
question = "Für wen ist das Buch laut Einleitung gedacht?"
retrieval_query = "programmieren lernen Zielgruppe Wissenschaft Wirtschaft Ingenieurwesen Landwirtschaft Finanzwesen Naturwissenschaften"

print("\n=== Frage ===")
print(question)

result = query_pipeline.run(
    data={
        "bm25_retriever": {
            "query": retrieval_query,
            "top_k": 5,
        },
        "text_embedder": {
            "text": retrieval_query,
        },
        "embedding_retriever": {
            "top_k": 5,
        },
        "prompt_builder": {
            "question": question,
        },
    },
    include_outputs_from={"joiner"},
)

print("\n=== Antwort ===")
print(result["llm"]["replies"][0].text)

print("\n=== Gefundene Chunks ===")
for i, doc in enumerate(result["joiner"]["documents"], start=1):
    page = doc.meta.get("page_number", "?")
    score = getattr(doc, "score", None)
    print(f"\nChunk {i} | Seite: {page} | Score: {score}")
    print(doc.content[:500])
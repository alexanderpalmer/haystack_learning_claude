from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

INDEX_PATH = "storage/buch_index"

# Zielordner anlegen
Path("storage").mkdir(exist_ok=True)

# Persistenter FAISS-Store
document_store = FAISSDocumentStore(
    index_path=INDEX_PATH,
    embedding_dim=1024,
)

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

print("=== Indexierung startet ===")
result = indexing_pipeline.run(
    {
        "converter": {
            "sources": ["data/buch.pdf"]
        }
    }
)

print(f"Geschriebene Dokumente: {result['writer']['documents_written']}")

# Persistenz explizit speichern
document_store.save(INDEX_PATH)

print("\n=== Fertig ===")
print(f"Index gespeichert unter: {INDEX_PATH}")
print("Erwartete Dateien:")
print(f"- {INDEX_PATH}.faiss")
print(f"- {INDEX_PATH}.json")
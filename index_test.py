from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

converter = PyPDFToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter(
    split_by="word",
    split_length=200,
    split_overlap=30
)
embedder = OllamaDocumentEmbedder(model="qwen3-embedding:0.6b")
writer = DocumentWriter(
    document_store=document_store,
    policy=DuplicatePolicy.OVERWRITE
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

result = indexing_pipeline.run({
    "converter": {
        "sources": ["data/buch.pdf"]
    }
})

print("=== Indexing-Test ===")
print(f"Geschriebene Dokumente: {result['writer']['documents_written']}")
print(f"Embedding-Metadaten: {result['embedder']['meta']}")
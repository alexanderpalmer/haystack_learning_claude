import re
from dataclasses import replace
from pathlib import Path
from typing import List

from pypdf import PdfReader

from haystack import Pipeline, component, Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.document_stores.faiss import FAISSDocumentStore


@component
class PerPagePDFConverter:
    """
    Extrahiert Text aus einer PDF-Datei seitenweise.
    Jede Seite wird zu einem eigenen Document mit korrekter page_number im Metadaten-Feld.
    """

    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]):
        documents = []
        for source in sources:
            reader = PdfReader(source)
            source_name = Path(source).name
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                doc = Document(
                    content=text,
                    meta={
                        "pdf_page": page_num,
                        "file_path": source_name,
                    },
                )
                documents.append(doc)
        return {"documents": documents}


@component
class PDFArtifactCleaner:
    """
    Bereinigt PDF-spezifische Artefakte:
    - Seitenumbruch-Zeichen (\x0c) mit folgender Kopfzeile
    - Digitales Wasserzeichen ("Persönliches Exemplar für ...")
    - Copyright-Footer ("© Rheinwerk Verlag, Bonn 2026")
    - Deutsche Silbentrennung über Zeilenumbrüche (z.B. "Kön-\nnen" → "können")
    - Punktketten in TOC-Zeilen kürzen, Kapitelname und Seitenzahl erhalten
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        cleaned = []
        for doc in documents:
            text = doc.content or ""

            # Seitenumbruch-Zeichen + nachfolgende Kopfzeile entfernen
            text = re.sub(r"\x0c[^\n]*\n?", " ", text)

            # Digitales Wasserzeichen entfernen
            text = re.sub(r"Persönliches Exemplar für[^\n]*\n?", "", text)

            # Copyright-Footer entfernen
            text = re.sub(r"© Rheinwerk[^\n]*", "", text)

            # Silbentrennung mit Leerzeichen vor Bindestrich: "ge -\ndruckten" → "gedruckten"
            text = re.sub(r"(\w+) -\n(\w)", r"\1\2", text)

            # Silbentrennung ohne Leerzeichen: "Kön-\nnen" → "können"
            text = re.sub(r"(\w+)-\n(\w)", r"\1\2", text)

            # Punktketten kürzen, Kapitelname und Seitenzahl erhalten
            # "Kapitel 3 .......... 42" → "Kapitel 3 42"
            text = re.sub(r"\.{4,}", " ", text)

            cleaned.append(replace(doc, content=text))
        return {"documents": cleaned}


INDEX_PATH = "storage/buch_index"

Path("storage").mkdir(exist_ok=True)

# Alten Index löschen damit keine Dokumente aus früheren Läufen akkumuliert werden
for ext in (".faiss", ".json"):
    p = Path(INDEX_PATH + ext)
    if p.exists():
        p.unlink()

document_store = FAISSDocumentStore(
    index_path=INDEX_PATH,
    embedding_dim=1024,
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PerPagePDFConverter())
indexing_pipeline.add_component("pdf_cleaner", PDFArtifactCleaner())
indexing_pipeline.add_component(
    "cleaner",
    DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
    ),
)
indexing_pipeline.add_component(
    "splitter",
    DocumentSplitter(
        split_by="word",
        split_length=300,
        split_overlap=50,
    ),
)
indexing_pipeline.add_component(
    "embedder",
    OllamaDocumentEmbedder(model="qwen3-embedding:0.6b", timeout=600),
)
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    ),
)

indexing_pipeline.connect("converter", "pdf_cleaner")
indexing_pipeline.connect("pdf_cleaner", "cleaner")
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

document_store.save(INDEX_PATH)

print("\n=== Fertig ===")
print(f"Index gespeichert unter: {INDEX_PATH}")
print(f"- {INDEX_PATH}.faiss")
print(f"- {INDEX_PATH}.json")

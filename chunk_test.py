from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

# PDF einlesen
converter = PyPDFToDocument()
result = converter.run(sources=["data/buch.pdf"])
documents = result["documents"]

# Text bereinigen
cleaner = DocumentCleaner()
cleaned_result = cleaner.run(documents=documents)
cleaned_documents = cleaned_result["documents"]

# In Chunks zerlegen
splitter = DocumentSplitter(
    split_by="word",
    split_length=200,
    split_overlap=30
)
split_result = splitter.run(documents=cleaned_documents)
chunks = split_result["documents"]

print("=== Chunk-Test ===")
print(f"Anzahl Chunks: {len(chunks)}")

if chunks:
    print("\n=== Erster Chunk ===")
    print(chunks[0].content)

    print("\n=== Metadaten des ersten Chunks ===")
    print(chunks[0].meta)

    if len(chunks) > 1:
        print("\n=== Zweiter Chunk ===")
        print(chunks[1].content)
else:
    print("Es wurden keine Chunks erzeugt.")
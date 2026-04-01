from haystack.components.converters import PyPDFToDocument

converter = PyPDFToDocument()

result = converter.run(sources=["data/buch.pdf"])
documents = result["documents"]

print("=== PDF-Test ===")
print(f"Anzahl erzeugter Dokumente: {len(documents)}")

if documents:
    first_doc = documents[0]
    print("\n=== Erster Textausschnitt ===")
    print(first_doc.content[:2000])
else:
    print("Es wurde kein Dokument aus dem PDF erzeugt.")
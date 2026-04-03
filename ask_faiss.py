import re

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

# Prompt: KONTEXT zuerst, dann FRAGE — verhindert, dass das Modell
# die Seiten-Abschnitte als mehrere Fragen interpretiert.
template = [
    ChatMessage.from_system(
        "Du bist ein präziser Assistent für das Buch "
        "'Python für KI- und Daten-Projekte' von Johannes Schildgen (Rheinwerk Verlag).\n\n"
        "STRENGE REGELN:\n"
        "1. Antworte IMMER auf Deutsch.\n"
        "2. Beantworte GENAU und NUR die gestellte Frage.\n"
        "3. Wenn nach einer Zahl, einem Titel oder einem kurzen Wert gefragt wird: "
        "Antworte NUR mit diesem Wert, ohne Erklärung oder Kommentar.\n"
        "4. Wenn eine ausführliche Erklärung gefragt wird: Antworte ausführlich und "
        "präzise, gib vollständige Code-Beispiele wieder.\n"
        "5. Stütze dich ausschließlich auf die bereitgestellten Kontext-Auszüge.\n"
        "6. Gib am Ende die Quell-Seitenzahlen an (z.B. 'Quellen: Seite 42, 43'), "
        "außer bei sehr kurzen Faktenfragen.\n"
        "7. Falls die Antwort nicht im Kontext steht: "
        "'Dazu enthält das Buch in den gefundenen Abschnitten keine Information.'"
    ),
    ChatMessage.from_user(
        "KONTEXT AUS DEM BUCH:\n"
        "====\n"
        "{% for doc in documents %}"
        "[Seite {{ doc.meta.pdf_page }}]\n"
        "{{ doc.content }}\n\n"
        "{% endfor %}"
        "====\n\n"
        "FRAGE: {{ question }}\n\n"
        "ANTWORT (auf Deutsch):"
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

# Strukturelle Schlüsselwörter → "Inhaltsverzeichnis" zur Embedding-Query hinzufügen
# damit TOC-Seiten bei Kapitel/Seiten-Fragen gefunden werden.
STRUCTURAL_PATTERNS = [
    r"\bkapitel\b", r"\bseite\b.*\bbeginnt\b", r"\bwie\s+viele\b",
    r"\bwas\s+ist\s+der\s+titel\b", r"\bwelches\s+thema\b",
    r"\bletztes?\s+kapitel\b", r"\berstes?\s+kapitel\b",
]


def expand_query(query: str) -> str:
    """Fügt 'Inhaltsverzeichnis' zu strukturellen Fragen hinzu."""
    q_lower = query.lower()
    if any(re.search(p, q_lower) for p in STRUCTURAL_PATTERNS):
        return f"{query} Inhaltsverzeichnis Kapitelübersicht"
    return query


question = input("Frage eingeben: ").strip()
search_query = expand_query(question)

print("\n=== Frage ===")
print(question)

result = query_pipeline.run(
    data={
        "text_embedder": {"text": search_query},
        "retriever": {"top_k": 15},
        "prompt_builder": {"question": question},
    },
    include_outputs_from={"retriever"},
)

print("\n=== Antwort ===")
print(result["llm"]["replies"][0].text)

print("\n=== Gefundene Chunks ===")
for i, doc in enumerate(result["retriever"]["documents"], start=1):
    page = doc.meta.get("pdf_page", "?")
    score = getattr(doc, "score", None)
    score_str = f"{score:.4f}" if score is not None else "n/a"
    print(f"\nChunk {i} | Seite: {page} | Score: {score_str}")
    print(doc.content[:300])

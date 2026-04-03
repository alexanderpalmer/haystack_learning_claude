"""
Automatisiertes Test-Skript für das RAG-System.
Führt 24 Fragen aus und bewertet die Qualität der Antworten.
"""
import re
import json
import time

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
query_pipeline.add_component("text_embedder", OllamaTextEmbedder(model="qwen3-embedding:0.6b"))
query_pipeline.add_component("retriever", FAISSEmbeddingRetriever(document_store=document_store))
query_pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=template, required_variables={"question", "documents"}),
)
query_pipeline.add_component(
    "llm",
    OllamaChatGenerator(model="llama3.2:3b", timeout=120, generation_kwargs={"temperature": 0.1, "num_predict": 512}),
)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "llm.messages")

STRUCTURAL_PATTERNS = [
    r"\bkapitel\b", r"\bseite\b.*\bbeginnt\b", r"\bwie\s+viele\b",
    r"\bwas\s+ist\s+der\s+titel\b", r"\bwelches\s+thema\b",
    r"\bletztes?\s+kapitel\b", r"\berstes?\s+kapitel\b",
]

# TOC-Seiten (5–10) direkt laden für Strukturfragen
toc_docs = sorted(
    [d for d in document_store.filter_documents() if 5 <= d.meta.get("pdf_page", 0) <= 10],
    key=lambda d: d.meta.get("pdf_page", 0),
)

llm = OllamaChatGenerator(
    model="llama3.2:3b", timeout=120, generation_kwargs={"temperature": 0.1, "num_predict": 512}
)


def is_structural(query: str) -> bool:
    q_lower = query.lower()
    return any(re.search(p, q_lower) for p in STRUCTURAL_PATTERNS)


def ask_structural(question: str) -> tuple[str, list, float]:
    """Beantwortet Strukturfragen direkt mit den TOC-Seiten."""
    # "Inhalt\n{Seitenzahl}\n" Header aus jedem Chunk entfernen
    clean_lines = []
    for d in toc_docs:
        for line in d.content.splitlines():
            stripped = line.strip()
            if stripped and stripped != "Inhalt" and not stripped.isdigit():
                clean_lines.append(stripped)
    context = "\n".join(clean_lines)

    # Nur Hauptkapitel (keine Unterabschnitte) für Zählfragen
    if re.search(r"\bwie\s+viele\b", question.lower()):
        chapters = [l for l in clean_lines if re.match(r"^\d{1,2}\s+[A-ZÄÖÜ]", l)]
        context = "\n".join(chapters)
    messages = [
        ChatMessage.from_system(
            "Du bist ein präziser Assistent. Antworte NUR mit dem gefragten Wert — "
            "keine Erklärung, kein Kommentar. Stütze dich ausschließlich auf den Kontext."
        ),
        ChatMessage.from_user(f"INHALTSVERZEICHNIS:\n====\n{context}\n====\n\nFRAGE: {question}\n\nANTWORT:"),
    ]
    result = llm.run(messages=messages)
    answer = result["replies"][0].text.strip()
    pages = sorted(set(d.meta["pdf_page"] for d in toc_docs))
    return answer, pages, 1.0


def expand_query(query: str) -> str:
    return query


# 24 Testfragen mit erwarteten Schlüsselbegriffen
test_questions = [
    # Strukturfragen (Inhaltsverzeichnis)
    {"nr": 1,  "frage": "Wie viele Kapitel hat das Buch? Nur die Zahl.",
               "erwartet": "12"},
    {"nr": 2,  "frage": "Auf welcher Seite beginnt Kapitel 7? Nur die Seitenzahl.",
               "erwartet": "147"},
    {"nr": 3,  "frage": "Was ist der Titel von Kapitel 8? Nur der Titel.",
               "erwartet": "KI in Aktion"},
    {"nr": 4,  "frage": "Welches Thema behandelt Kapitel 5? Nur den Titel.",
               "erwartet": "Datenanalysen"},
    {"nr": 5,  "frage": "Auf welcher Seite beginnt Kapitel 11? Nur die Seitenzahl.",
               "erwartet": "263"},
    {"nr": 6,  "frage": "Wie heißt das letzte Kapitel des Buches? Nur den Titel.",
               "erwartet": "Routineaufgaben"},

    # Python-Grundlagen (Kapitel 3)
    {"nr": 7,  "frage": "Was ist der Walrus-Operator in Python und wie wird er eingesetzt?",
               "erwartet": ":="},
    {"nr": 8,  "frage": "Was ist Tupelzerlegung (Tuple Unpacking) in Python?",
               "erwartet": "Tupel"},
    {"nr": 9,  "frage": "Wie erstellt man eine virtuelle Umgebung mit venv in Python?",
               "erwartet": "venv"},
    {"nr": 10, "frage": "Was ist der Unterschied zwischen Liste, Menge (Set) und Tupel in Python?",
               "erwartet": "Liste"},

    # Dateien (Kapitel 4)
    {"nr": 11, "frage": "Wie liest man eine Textdatei in Python? Zeige ein Code-Beispiel.",
               "erwartet": "open"},
    {"nr": 12, "frage": "Wie liest man JSON-Dateien in Python?",
               "erwartet": "json"},

    # Datenanalyse (Kapitel 5)
    {"nr": 13, "frage": "Was ist ein Pandas DataFrame?",
               "erwartet": "DataFrame"},
    {"nr": 14, "frage": "Wie liest man eine CSV-Datei mit Pandas ein?",
               "erwartet": "read_csv"},
    {"nr": 15, "frage": "Wie entfernt man Duplikate in einem Pandas DataFrame?",
               "erwartet": "drop_duplicates"},

    # Visualisierung (Kapitel 6)
    {"nr": 16, "frage": "Wie erstellt man ein Balkendiagramm mit Matplotlib?",
               "erwartet": "bar"},

    # Machine Learning (Kapitel 7)
    {"nr": 17, "frage": "Was ist lineare Regression und wofür wird sie eingesetzt?",
               "erwartet": "Regression"},
    {"nr": 18, "frage": "Was ist KNN (k-Nearest Neighbors)?",
               "erwartet": "Nachbarn"},
    {"nr": 19, "frage": "Was ist Clustering beim unsupervised Machine Learning?",
               "erwartet": "Cluster"},

    # KI Text/Bild (Kapitel 8)
    {"nr": 20, "frage": "Was ist Sentiment-Analyse?",
               "erwartet": "Sentiment"},
    {"nr": 21, "frage": "Was ist Named-Entity Recognition (NER)?",
               "erwartet": "Entit"},
    {"nr": 22, "frage": "Was ist Transfer Learning?",
               "erwartet": "Transfer"},

    # Web & APIs (Kapitel 9/10)
    {"nr": 23, "frage": "Wie macht man eine HTTP-GET-Anfrage mit der requests-Bibliothek in Python?",
               "erwartet": "requests.get"},

    # Datenbanken (Kapitel 11)
    {"nr": 24, "frage": "Wie greift man mit Python auf eine SQLite-Datenbank zu?",
               "erwartet": "sqlite3"},
]

results = []
total = len(test_questions)
correct = 0

print(f"=== RAG Test-Suite: {total} Fragen ===\n")

for item in test_questions:
    nr = item["nr"]
    question = item["frage"]
    search_query = expand_query(question)
    print(f"[{nr:02d}/{total}] {question[:70]}...")

    answer, pages, top_score = None, [], 0.0
    for attempt in range(1, 4):
        try:
            if is_structural(question):
                answer, pages, top_score = ask_structural(question)
            else:
                result = query_pipeline.run(
                    data={
                        "text_embedder": {"text": search_query},
                        "retriever": {"top_k": 5},
                        "prompt_builder": {"question": question},
                    },
                    include_outputs_from={"retriever"},
                )
                answer = result["llm"]["replies"][0].text.strip()
                chunks = result["retriever"]["documents"]
                pages = sorted(set(d.meta.get("pdf_page", "?") for d in chunks))
                top_score = max((getattr(d, "score", 0) or 0) for d in chunks)
            break
        except Exception as e:
            wait = 10 * attempt
            print(f"  [Versuch {attempt}/3] Fehler: {e!s:.120} — warte {wait}s ...")
            time.sleep(wait)
            if attempt == 3:
                answer = f"FEHLER: {e}"

    # Einfache Keyword-Prüfung ob Schlüsselbegriff in der Antwort vorkommt
    expected_kw = item["erwartet"]
    hit = expected_kw.lower() in answer.lower()
    if hit:
        correct += 1

    status = "✓" if hit else "✗"
    entry = {
        "nr": nr,
        "frage": question,
        "erwartet_kw": expected_kw,
        "antwort": answer,
        "seiten": pages,
        "top_score": round(top_score, 4),
        "hit": hit,
    }
    results.append(entry)

    print(f"  {status} Erwartet: {expected_kw!r:30s} | Score: {top_score:.4f} | Seiten: {pages[:5]}")
    print(f"      Antwort: {answer[:200]}")
    print()

# Ergebnisse speichern
out_path = "storage/test_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

pct = correct / total * 100
print(f"\n{'='*60}")
print(f"=== ERGEBNIS: {correct}/{total} korrekt ({pct:.0f}%) ===")
print(f"=== Ergebnisse gespeichert: {out_path} ===")
print(f"{'='*60}")

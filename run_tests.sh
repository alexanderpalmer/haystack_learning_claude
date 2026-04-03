#!/bin/bash
# Führt 24 Testfragen aus und speichert jede Antwort einzeln
set -e
source .venv/bin/activate
mkdir -p storage/test_answers

questions=(
    "Wie viele Kapitel hat das Buch? Nur die Zahl."
    "Auf welcher Seite beginnt Kapitel 7? Nur die Seitenzahl."
    "Was ist der Titel von Kapitel 8? Nur der Titel."
    "Welches Thema behandelt Kapitel 5? Nur den Titel."
    "Auf welcher Seite beginnt Kapitel 11 Datenbanken? Nur die Seitenzahl."
    "Wie heißt das letzte Kapitel des Buches? Nur den Titel."
    "Was ist der Walrus-Operator in Python?"
    "Was ist Tupelzerlegung in Python?"
    "Wie erstellt man eine virtuelle Umgebung mit venv?"
    "Was ist der Unterschied zwischen Liste, Menge und Tupel in Python?"
    "Wie liest man eine Textdatei in Python? Zeige ein Beispiel."
    "Wie liest man JSON-Dateien in Python?"
    "Was ist ein Pandas DataFrame?"
    "Wie liest man eine CSV-Datei mit Pandas ein?"
    "Wie entfernt man Duplikate in einem Pandas DataFrame?"
    "Wie erstellt man ein Balkendiagramm mit Matplotlib?"
    "Was ist lineare Regression und wofür wird sie eingesetzt?"
    "Was ist KNN (k-Nearest Neighbors)?"
    "Was ist Clustering beim Machine Learning?"
    "Was ist Sentiment-Analyse?"
    "Was ist Named-Entity Recognition (NER)?"
    "Was ist Transfer Learning?"
    "Wie macht man eine HTTP-GET-Anfrage mit requests in Python?"
    "Wie greift man mit Python auf eine SQLite-Datenbank zu?"
)

total=${#questions[@]}
echo "=== RAG Test-Suite: $total Fragen ===" > storage/test_log.txt

for i in "${!questions[@]}"; do
    nr=$((i+1))
    q="${questions[$i]}"
    echo "[${nr}/${total}] ${q:0:70}..."
    echo "[${nr}/${total}] ${q}" >> storage/test_log.txt

    python -u -c "
import sys
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

ds = FAISSDocumentStore(index_path='storage/buch_index', embedding_dim=1024)
template = [
    ChatMessage.from_system(
        \"Du bist ein Assistent fuer das Buch 'Python fuer KI- und Daten-Projekte' von Johannes Schildgen.\"
        \" Beantworte die Frage praezise anhand der Buchauszuege.\"
        \" Passe die Laenge an: kurze Faktenfragen -> kurze Antwort, erklaerende Fragen -> ausfuehrlich.\"
        \" Wenn die Antwort nicht im Kontext ist: 'Nicht im Kontext gefunden.'\"
    ),
    ChatMessage.from_user(
        'Frage: {{ question }}\n\nBuchauszuege:\n'
        '{% for doc in documents %}--- Seite {{ doc.meta.page_number }} ---\n{{ doc.content }}\n\n{% endfor %}'
        'Beantworte die Frage basierend auf den obigen Auszuegen.'
    ),
]
p = Pipeline()
p.add_component('emb', OllamaTextEmbedder(model='qwen3-embedding:0.6b'))
p.add_component('ret', FAISSEmbeddingRetriever(document_store=ds))
p.add_component('pb', ChatPromptBuilder(template=template, required_variables={'question','documents'}))
p.add_component('llm', OllamaChatGenerator(model='gemma3:4b', timeout=300, generation_kwargs={'temperature':0.1}))
p.connect('emb.embedding','ret.query_embedding')
p.connect('ret.documents','pb.documents')
p.connect('pb.prompt','llm.messages')

question = sys.argv[1]
r = p.run({'emb':{'text':question},'ret':{'top_k':15},'pb':{'question':question}}, include_outputs_from={'ret'})
ans = r['llm']['replies'][0].text.strip()
pages = sorted(set(d.meta.get('page_number','?') for d in r['ret']['documents']))
scores = [round(getattr(d,'score',0) or 0,4) for d in r['ret']['documents']]
print(ans)
print('SEITEN:', pages)
print('TOP_SCORE:', max(scores) if scores else 0)
" "$q" 2>/dev/null > "storage/test_answers/q${nr}.txt"

    echo "Antwort: $(head -1 storage/test_answers/q${nr}.txt)" >> storage/test_log.txt
    echo "$(head -1 storage/test_answers/q${nr}.txt)"
    echo "---"
done

echo "=== Fertig ===" | tee -a storage/test_log.txt

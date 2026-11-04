# KI-Bildanalyse 1.0

Native macOS-Oberfläche nach der React/pywebview-Architektur aus Use Case 1. Das vorhandene Modul `Tool/ki_bildanalyse.py` bleibt die einzige Klassifikationslogik.

## Entwicklung starten

```bash
npm --prefix frontend install
npm --prefix frontend run build
python3 app.py
```

Für Frontend-Entwicklung mit Hot Reload:

```bash
npm --prefix frontend run dev
python3 app.py --dev-url http://127.0.0.1:5173
```

## macOS-App bauen

```bash
python3 -m pip install -r requirements-build.txt
chmod +x build_macos.sh
./build_macos.sh
```

Das App-Bundle wird unter `dist/KI-Bildanalyse.app` erstellt. Qwen3-VL verwendet die lokale Ollama-Instanz. Für neue lokale Analysen wird `qwen3-vl:4b-instruct` verwendet, weil die separate Thinking-Variante ihren Denkmodus nicht per `think: false` abschalten kann. Der Request dokumentiert trotzdem ausdrücklich `think: false`, `num_ctx=24576` und `num_predict=4096`. Historische Ergebnisse anderer Modelle bleiben lesbar. OpenAI und Mistral werden nur nach zusätzlicher Kostenbestätigung in der Oberfläche aufgerufen; die CLI verlangt weiterhin `--live --confirm-paid-request`. Der Mistral-Key wird ausschließlich aus dem macOS-Schlüsselbunddienst `Usecase_2_Mistral` gelesen.

Projektmetadaten, Ergebnisse, fachliche Korrekturen und Exporte liegen getrennt von Originalbildern unter `Output/EKI_Bildanalyse/Projekte/`.

Die Oberfläche unterstützt unabhängige Mehrmodell- und Batchanalysen. Prüfkörper und Modelle werden per Checkbox kombiniert; die Aufträge laufen auf dem lokalen Mac sequenziell. Bis zu drei historische Läufe eines Prüfkörpers lassen sich nebeneinander vergleichen und getrennt fachlich bewerten. OpenAI- und Mistral-Aufträge benötigen vor dem Start eine gemeinsame Kosten- und Bildübertragungsfreigabe mit der angezeigten Zahl geplanter Cloud-Anfragen.

Beim ersten Öffnen eines Prüfkörpers sind höchstens drei über die Bildserie verteilte Schadensfotos vorausgewählt. Nummernfotos bleiben ausgeschlossen; alle vorhandenen Bilder werden in der Galerie angezeigt und können manuell ausgewählt werden. Laufende Aufträge zeigen Modell, Status, ausgewählte Dateinamen und eine sekündlich aktualisierte Laufzeit. Ollama erhält bis zu 15 Minuten für eine Antwort; Zeitüberschreitungen werden als verständlicher Fehler am Auftrag angezeigt.

Die Hauptoberfläche ist in die vier Register `Bildanalyse`, `Ergebnisse`, `Referenzdatenbank` und `Export` gegliedert. Einzelne, mehrere oder alle tatsächlich vorhandenen Prüfkörper werden in `Bildanalyse` über denselben wissenschaftlichen Variantenpfad geplant und gestartet. Identifikations- beziehungsweise Nummernfotos sind in der Galerie sichtbar, aber nicht aktivierbar und werden nie an ein Modell übertragen. KI-Beobachtungen, unverbindliche Klassifikationskandidaten und die separat gespeicherte fachliche Bewertung werden deutlich getrennt dargestellt.

Die zwölf aktiven Merkmalscodes GSF, LSF, KA, SA, sp HB, Spl HB, RiSch, LR, Bu, AA, SSA und KB sind für Evaluationsvariante 1 freigegeben. Der Sicherheitsmodus öffnet dabei kein Referenzmanifest, erzwingt null übertragene Referenzbilder und verlangt vor jedem Lauf einen aktuellen Plan. Dieser weist Bildbestätigungen, Variantenstatus, Kostenhinweis und echte aktive Auftragskonflikte aus. KB wird als sichtbare Prüf- beziehungsweise Einspannspur und nicht als Holzmerkmal behandelt.

Neue Ergebnisse kennzeichnen jeden Klassifikationsvorschlag zusätzlich als „Sicher sichtbar“ oder „Vermutet“. Diese Evidenzkennzeichnung wird in der Ergebnisansicht angezeigt; die bisherige Konfidenz bleibt davon getrennt und unverändert erhalten. Alle zwölf Merkmale bleiben gleichberechtigt, Mehrfachklassifikationen vollständig sichtbar und eine optionale Primärempfehlung ist keine nachgewiesene Bruchursache.

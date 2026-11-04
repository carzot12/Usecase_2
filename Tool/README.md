# KI-Bildanalyse – Prototyp

Der Prototyp analysiert die ausgewählten Fotos eines **unbekannten** Holzprüfkörpers. Standardmäßig läuft er ausschließlich mit einer lokalen Mock-Antwort und führt keinen API-Aufruf aus. Die Desktop-App wählt beim ersten Öffnen bis zu drei über die Bildserie verteilte Schadensfotos aus; alle Bilder bleiben in der Galerie manuell auswählbar. Nummernfotos werden nicht als Analysebilder übertragen. Während eines Batch-Laufs zeigt die App Modell, Bearbeitungsstatus, ausgewählte Dateinamen und die verstrichene Laufzeit.

## Sicherheits- und Wissenschaftsgrenzen

- Zulässig sind nur die 29 Codes aus `Datenbank/versagensarten_erweitert.json`.
- Zwölf Merkmalscodes besitzen vollständig ausgearbeitete Regeln in `klassifikationsregeln.json`. KB wird als sichtbare Prüf- beziehungsweise Einspannspur und nicht als Holzmerkmal behandelt.
- Als Bildbeispiele werden ausschließlich bestätigte Hauptreferenzbilder mit `default_for_inference=true` verwendet. Kontextbilder werden nie als Beispiele übertragen. Standardmäßig wird zur Kostenbegrenzung ein Hauptreferenzbild je belegter Klasse gesendet.
- Eingabebilder werden anhand von Pfad und SHA-256 gegen bekannte Referenz- und Kontextbilder geprüft.
- Versuchsprotokolle und dokumentierte Bruchursachen werden nicht geladen oder übertragen.
- Sichtbare Merkmale sind keine nachgewiesenen Versagensursachen.
- Originalbilder werden nur gelesen und nicht verändert.
- Für die Übertragung werden verkleinerte JPEG-Kopien ausschließlich im Arbeitsspeicher erzeugt (maximal 2048 Pixel lange Kante).

## Eingabe

Lege die zu analysierenden Dateien (`jpg`, `jpeg`, `png`, `webp` oder nicht animiertes `gif`) im Ordner des Prüfkörpers ab. Der CLI-Aufruf verarbeitet alle unterstützten Dateien; die Desktop-App überträgt ausschließlich die angehakten Bilder. Die Ergebnisdatei dokumentiert einbezogene und ausgeschlossene Dateinamen.

```text
Pruefdaten/Neue_Analysen/Test_001/
```

## Lokaler Mock-Test

Vom Ordner `Usecase_2` aus:

```bash
python3 -m Tool.ki_bildanalyse Pruefdaten/Neue_Analysen/Test_001
```

Die mitgelieferte Mock-Antwort liefert absichtlich `UNKLAR`. Ergebnisse werden unter `Output/KI_Klassifikation/` gespeichert.

## Lokale Ollama-Analyse

Ollama wird ausschließlich über `http://localhost:11434/api/chat` angesprochen. Dafür sind weder `--live` noch `--confirm-paid-request` erforderlich. Neue Analysen nutzen bei allen Anbietern dieselben zwölf aktiven Klassen, Regeln, Referenzbilder, Testbilder und dasselbe Ergebnisschema. Die Definitionen aller 29 Klassen bleiben für spätere Erweiterungen erhalten.

```bash
python3 -m Tool.ki_bildanalyse Pruefdaten/Neue_Analysen/Test_001/Sortiert/Z-30-12 \
  --provider ollama \
  --model qwen3-vl:4b-instruct \
  --ollama-num-ctx 24576 \
  --ollama-max-image-edge 768 \
  --max-references-per-class 1 \
  --output-dir Output/KI_Klassifikation/Testlauf_5_Pruefkoerper/Ollama_Ergebnisse
```

Qwen3-VL ist das einzige in der Desktop-App angebotene lokale Visionmodell. `--ollama-num-ctx` steuert die lokale Kontextgröße und steht standardmäßig auf 24576 Tokens. `--ollama-num-predict` steuert davon getrennt das maximale Generationsbudget und steht standardmäßig auf 4096 Tokens. Um die vielen variablen Vision-Tokens ressourcenschonend zu begrenzen, werden nur die temporären Ollama-Kopien standardmäßig auf maximal 768 Pixel Kantenlänge skaliert; anpassbar über `--ollama-max-image-edge`. Originalbilder und OpenAI-Bildauflösung bleiben unverändert. Ergebnisdateien enthalten eine eindeutige Run-ID, Projekt und Prüfkörper, Anbieter, angeforderte und tatsächlich gemeldete Modellkennung, Laufzeit, Prompt-Version und -Hash, Bild- und Referenzhashes sowie die Provider-Metadaten. Historische Ergebnisdateien mit früheren Modellkennungen bleiben lesbar.

## Unabhängige Modellvergleiche und Batch-Auswertung

Die Desktop-App kann mehrere Prüfkörper und Modelle kombinieren. Jeder Auftrag baut seinen Prompt neu aus den festgelegten Bildern, Definitionen, Regeln und Referenzen auf. Frühere KI-Ergebnisse und fachliche Bewertungen werden nicht als Modellinput gelesen. Batch-Aufträge laufen standardmäßig sequenziell und werden unter `Output/EKI_Bildanalyse/Projekte/<Projekt>/Batches/` gespeichert. Erfolgreiche Aufträge werden beim Fortsetzen nicht erneut ausgeführt oder überschrieben.

Fachliche Bewertungen liegen getrennt unter `Bewertungen/`. Historische Ergebnisdateien bleiben lesbar; neue Dateien verwenden Schema 4.1 mit `observations_short`, `visible_features`, gleichrangigen `class_suggestions` und `uncertain_points`. Jeder Klassenvorschlag trägt zusätzlich zu der unveränderten Konfidenzstufe (`hoch`, `mittel`, `unsicher`) den Evidenzstatus `sicher_sichtbar` oder `vermutet`. Die optionale Primärempfehlung ist nur ein visueller Vorschlag; eine KI-seitige Hauptversagensursache wird nicht gespeichert. Die Ergebnisdateien protokollieren weiterhin die für Reproduzierbarkeit erforderlichen Hashes und Einstellungen. Cloud-Aufträge starten nur nach gemeinsamer Bestätigung der angezeigten Anzahl geplanter Requests und der Bildübertragung.

## Erster freigegebener Live-Test

Der dokumentierte Kandidat ist `gpt-6.1-sol`. Die Modellprüfung ist in `model_verification.json` festgehalten; der konkrete Kontozugriff ist ohne API-Anfrage nicht überprüfbar.

```bash
USECASE2_OPENAI_MODEL=gpt-6.1-sol python3 -m Tool.ki_bildanalyse Pruefdaten/Neue_Analysen/Test_001 --provider openai --live --confirm-paid-request
```

Der Live-Aufruf ist nur möglich, wenn beide Flags gesetzt sind. Der Schlüssel wird zuerst aus `OPENAI_API_KEY`, andernfalls aus dem macOS-Schlüsselbunddienst `KI-Bemessungstool OpenAI` gelesen und niemals protokolliert.

Mit `--max-references-per-class 2` können nach einer Kostenabwägung zwei bestätigte Beispiele je belegter Klasse übertragen werden.

## Mistral Vision

Mistral verwendet dieselben Testbilder, bestätigten Hauptreferenzbilder, zwölf aktiven Klassen,
Klassifikationsregeln und dasselbe Ergebnisschema wie OpenAI. Das Standardmodell ist
`ministral-3b-2512`. Ohne `--live` wird nur die lokale Mock-Pipeline ausgeführt; dabei
werden weder der Schlüsselbund gelesen noch Netzwerkdaten übertragen.

```bash
python3 -m Tool.ki_bildanalyse \
  Pruefdaten/Neue_Analysen/Test_001/Sortiert/Z-30-12 \
  --provider mistral \
  --model ministral-3b-2512 \
  --max-references-per-class 1 \
  --output-dir Output/KI_Klassifikation/Testlauf_5_Pruefkoerper/Mistral_Mock_Ergebnisse
```

Ein späterer echter Aufruf benötigt beide Kostenfreigabe-Flags:

```bash
python3 -m Tool.ki_bildanalyse <PRUEFKOERPERORDNER> \
  --provider mistral --model ministral-3b-2512 \
  --live --confirm-paid-request
```

Der API-Schlüssel wird ausschließlich aus dem macOS-Schlüsselbunddienst
`Usecase_2_Mistral` gelesen. Es gibt für Mistral bewusst keinen Fallback auf eine
Umgebungsvariable. Ergebnisdateien dokumentieren Anbieter, Modell, Laufzeit,
Tokenverbrauch und – soweit die API ihn liefert – Kosteninformationen getrennt.
Wenn mehr Bilder ausgewählt sind, als eine Anfrage aufnehmen kann, werden sie in
Teilgruppen verarbeitet; die Ergebnis-Metadaten listen die verwendeten Bilder je
Teilgruppe auf.

### Bekannter Mistral-Fehlerstand

Ein früherer Live-Versuch mit `ministral-3b-2512` endete mit HTTP 422, weil der
direkte HTTP-Payload den SDK-internen Feldnamen `schema_definition` statt des
API-Felds `schema` enthielt. Der lokale Adapter und die Regressionstests sind
auf `json_schema.schema` umgestellt. Da seit dieser Korrektur bewusst kein
kostenpflichtiger Live-Aufruf ausgeführt wurde, ist die Behebung gegenüber der
Mistral-Produktiv-API noch nicht live verifiziert. Die Live-Sperre mit
`--live --confirm-paid-request` bleibt unverändert bestehen.

## Evaluationsvariante 1

Der Schalter `--evaluation-variant 1` aktiviert eine harte referenzfreie Grenze. Das Referenzmanifest wird in diesem Modus nicht geöffnet, und die Ergebnisdatei protokolliert `reference_database_accessed=false` sowie null verwendete Referenzbilder. Die zwölf aktiven Codes GSF, LSF, KA, SA, sp HB, Spl HB, RiSch, LR, Bu, AA, SSA und KB werden dynamisch aus `Datenbank/klassifikationsregeln.json` übernommen; künftig dort ergänzte dokumentierte Klassen benötigen keine fest codierte Anzahl. Pro Lauf werden Regelversion und SHA-256 der tatsächlich verwendeten kompakten Klassenbeschreibungen protokolliert. Fachliche Prüfvermerke, Literaturquellen, offene Forschungsfragen, Ground Truth, Expertenbewertungen und frühere KI-Ergebnisse werden nicht in den Modellprompt übernommen.

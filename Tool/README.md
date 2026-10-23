# KI-Bildanalyse – Prototyp

Der Prototyp analysiert 1–3 Fotos eines **unbekannten** Holzprüfkörpers. Standardmäßig läuft er ausschließlich mit einer lokalen Mock-Antwort und führt keinen API-Aufruf aus.

## Sicherheits- und Wissenschaftsgrenzen

- Zulässig sind nur die 29 Codes aus `Datenbank/versagensarten_erweitert.json`.
- Die sechs detailliert geprüften Regeln aus `klassifikationsregeln.json` ergänzen die Klassendefinitionen.
- Als Bildbeispiele werden ausschließlich bestätigte Hauptreferenzbilder mit `default_for_inference=true` verwendet. Kontextbilder werden nie als Beispiele übertragen. Standardmäßig wird zur Kostenbegrenzung ein Hauptreferenzbild je belegter Klasse gesendet.
- Eingabebilder werden anhand von Pfad und SHA-256 gegen bekannte Referenz- und Kontextbilder geprüft.
- Versuchsprotokolle und dokumentierte Bruchursachen werden nicht geladen oder übertragen.
- Sichtbare Merkmale sind keine nachgewiesenen Versagensursachen.
- Originalbilder werden nur gelesen und nicht verändert.
- Für die Übertragung werden verkleinerte JPEG-Kopien ausschließlich im Arbeitsspeicher erzeugt (maximal 2048 Pixel lange Kante).

## Eingabe

Lege genau 1–3 Dateien (`jpg`, `jpeg`, `png`, `webp` oder nicht animiertes `gif`) hier ab:

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

Ollama wird ausschließlich über `http://localhost:11434/api/chat` angesprochen. Dafür sind weder `--live` noch `--confirm-paid-request` erforderlich. Die Analyse nutzt dieselben 29 Klassen, Regeln, Referenzbilder, Testbilder und dasselbe Ergebnisschema wie OpenAI.

```bash
python3 -m Tool.ki_bildanalyse Pruefdaten/Neue_Analysen/Test_001/Sortiert/Z-30-12 \
  --provider ollama \
  --model llava:latest \
  --ollama-num-ctx 16384 \
  --ollama-max-image-edge 768 \
  --max-references-per-class 1 \
  --output-dir Output/KI_Klassifikation/Testlauf_5_Pruefkoerper/Ollama_Ergebnisse
```

Andere lokal installierte Visionmodelle können beispielsweise mit `--model qwen3-vl:4b` gewählt werden. `--ollama-num-ctx` steuert die lokale Kontextgröße; der Standardwert ist 16384 Tokens. Größere Werte erhöhen den Arbeitsspeicherbedarf. Um die vielen variablen Qwen-Vision-Tokens ressourcenschonend zu begrenzen, werden nur die temporären Ollama-Kopien standardmäßig auf maximal 768 Pixel Kantenlänge skaliert; anpassbar über `--ollama-max-image-edge`. Der interne Qwen-Denkmodus wird für die streng strukturierte Klassifikation mit `think: false` angefordert; die installierte Qwen-Variante kann dennoch interne Thinking-Tokens erzeugen, weshalb ausreichend freier Kontext benötigt wird. Originalbilder und OpenAI-Bildauflösung bleiben unverändert. Ergebnisdateien enthalten Anbieter, Modell, Gesamtlaufzeit, Ollama-Metadaten einschließlich der tatsächlich gesendeten Request-Optionen und die validierte Klassifikation im Dateinamen und im JSON-Dokument.

## Erster freigegebener Live-Test

Der dokumentierte Kandidat ist `gpt-6.1-sol`. Die Modellprüfung ist in `model_verification.json` festgehalten; der konkrete Kontozugriff ist ohne API-Anfrage nicht überprüfbar.

```bash
USECASE2_OPENAI_MODEL=gpt-6.1-sol python3 -m Tool.ki_bildanalyse Pruefdaten/Neue_Analysen/Test_001 --provider openai --live --confirm-paid-request
```

Der Live-Aufruf ist nur möglich, wenn beide Flags gesetzt sind. Der Schlüssel wird zuerst aus `OPENAI_API_KEY`, andernfalls aus dem macOS-Schlüsselbunddienst `KI-Bemessungstool OpenAI` gelesen und niemals protokolliert.

Mit `--max-references-per-class 2` können nach einer Kostenabwägung zwei bestätigte Beispiele je belegter Klasse übertragen werden.

## Mistral Vision

Mistral verwendet dieselben Testbilder, bestätigten Hauptreferenzbilder, 29 Klassen,
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

### Bekannter Mistral-Fehlerstand

Ein früherer Live-Versuch mit `ministral-3b-2512` endete mit HTTP 422, weil der
direkte HTTP-Payload den SDK-internen Feldnamen `schema_definition` statt des
API-Felds `schema` enthielt. Der lokale Adapter und die Regressionstests sind
auf `json_schema.schema` umgestellt. Da seit dieser Korrektur bewusst kein
kostenpflichtiger Live-Aufruf ausgeführt wurde, ist die Behebung gegenüber der
Mistral-Produktiv-API noch nicht live verifiziert. Die Live-Sperre mit
`--live --confirm-paid-request` bleibt unverändert bestehen.


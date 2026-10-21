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

## Erster freigegebener Live-Test

Der dokumentierte Kandidat ist `gpt-6.1-sol`. Die Modellprüfung ist in `model_verification.json` festgehalten; der konkrete Kontozugriff ist ohne API-Anfrage nicht überprüfbar.

```bash
USECASE2_OPENAI_MODEL=gpt-6.1-sol python3 -m Tool.ki_bildanalyse Pruefdaten/Neue_Analysen/Test_001 --live --confirm-paid-request
```

Der Live-Aufruf ist nur möglich, wenn beide Flags gesetzt sind. Der Schlüssel wird zuerst aus `OPENAI_API_KEY`, andernfalls aus dem macOS-Schlüsselbunddienst `KI-Bemessungstool OpenAI` gelesen und niemals protokolliert.

Mit `--max-references-per-class 2` können nach einer Kostenabwägung zwei bestätigte Beispiele je belegter Klasse übertragen werden.


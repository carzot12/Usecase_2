# Reproduzierbare Evaluation der KI-Bildanalyse

## Datenbasis und Aussagegrenze

`ground_truth_manifest.json` enthält ausschließlich die 26 manuell bestätigten
Hauptreferenzbilder aus der Referenzbilddatenbank. Unbestätigte Kontextbilder,
Protokoll-Bruchursachen, KI-Vorschläge und frühere Modellergebnisse werden nicht
als Ground Truth verwendet.

Der unveränderte Datensatz deckt aktuell 11 von 29 Klassen ab. Neue
Bildanalysen verwenden die zwölf gleichwertigen Merkmalscodes GSF, LSF, KA, SA,
sp HB, Spl HB, RiSch, LR, Bu, AA, SSA und KB aus der zentralen Konfiguration. Die
Ground-Truth-Abdeckung bleibt davon getrennt und kann je Code unvollständig
sein. Es handelt sich um eine **interne
Leave-one-Prüfkörper-out-Evaluation**, nicht um eine unabhängige externe Validierung.
Für jeden Fall werden sämtliche Referenzen desselben Prüfkörpers und identische
Bilddateien ausgeschlossen. Bei `Bu`, `HB` und `RiSch` gibt es nur je einen
bestätigten Prüfkörper; in dessen Holdout steht daher keine Referenz derselben
Klasse zur Verfügung.

## Analysevarianten

- A: nur Testbilder und erlaubte Klassencodes
- B: Testbilder plus Beschreibungen der zwölf aktiven Merkmale
- C: Testbilder plus bestätigte Referenzbilder, ohne Klassenbeschreibungen
- D: zwei getrennte Modellaufrufe: zuerst Merkmalsextraktion ohne Klassenwissen,
  danach regelgestützte Klassifikation ausschließlich aus diesen Beobachtungen;
  keine Referenzbilder

Alle vier Varianten verwenden pro Lauf denselben Prüfkörpersatz, denselben
Provider, dieselbe Modellkennung und die dokumentierten Providerparameter.
Der Laufplan fixiert zusätzlich SHA-256-Prüfsummen von Ground Truth,
Evaluationscode, Klassendefinitionen, Regeln und Referenzmanifest.
Ein vollständiger Lauf umfasst 90 Modellaufrufe: je 18 für A, B und C sowie
36 für die zweistufige Variante D. Das ist insbesondere vor Cloudläufen bei der
Kostenabschätzung zu berücksichtigen.

## Kostenfreier Infrastrukturtest

```bash
cd ~/Desktop/Masterarbeit/Usecase_2
python3 -m Tool.evaluation prepare --validate-only
python3 -m Tool.evaluation run \
  --provider ollama \
  --model qwen3-vl:4b
```

Ohne `--execute` wird ausschließlich ein neuer, unveränderlicher Laufplan unter
`Output/KI_Evaluation/` angelegt. Es findet kein Modell- oder Netzwerkaufruf statt.

## Spätere Ausführung

Ein lokaler Ollama-Lauf wird ausdrücklich mit `--execute` gestartet:

```bash
python3 -m Tool.evaluation run \
  --provider ollama \
  --model qwen3-vl:4b \
  --ollama-num-ctx 16384 \
  --execute
```

Cloudläufe bleiben dreifach abgesichert: `--execute`, `--live` und
`--confirm-paid-request` sind gemeinsam erforderlich. Ohne ausdrückliche
Freigabe dürfen diese Befehle nicht ausgeführt werden.

```bash
python3 -m Tool.evaluation run \
  --provider openai \
  --model <verifizierte-Modellkennung> \
  --execute --live --confirm-paid-request
```

Jeder ausgeführte Lauf schreibt `evaluation_results.json`,
`evaluation_results.csv` und `confusion_matrices.csv` in einen neuen Ordner.
Fehler werden pro Einzelfall als `wrong_classification`, `unclassifiable`,
`invalid_json` oder `api_error` getrennt erfasst; ein einzelner Fehler bricht
den Gesamtlauf nicht ab.

Die Hauptauswertung (`metrics_by_variant`) berücksichtigt sämtliche
Klassifikationsvorschläge, also `sicher_sichtbar` und `vermutet`. Daneben enthält
`secure_visible_metrics_by_variant` eine getrennte Auswertung nur der sicher
sichtbaren Vorschläge. Für diese Zusatzsicht werden Abdeckungsrate,
ausgeschlossene beziehungsweise nicht bewertete Fälle, die Genauigkeit der
bewerteten Teilmenge und die effektive Trefferquote über alle Fälle gemeinsam
ausgewiesen. Dadurch darf der Ausschluss vermuteter Vorschläge nicht als
künstliche Verbesserung der Güte gelesen werden.

## Methodische Hinweise

- Trefferquote und Macro-F1 gelten nur für die im Datensatz vertretenen Klassen.
- `UNKLAR`, ungültige Antworten und API-Fehler zählen in der Gesamtquote als
  nicht korrekt und werden zusätzlich separat ausgewiesen.
- Eine belastbare Aussage über alle 29 Klassen erfordert weitere fachlich
  bestätigte, vom Referenzpool unabhängige Prüfkörper.
- Formal fachlich validierte Regeln liegen derzeit nur für 6 Klassen vor. Bei
  den übrigen 23 Klassen verwendet Variante D die erweiterten, teils als
  prüfbedürftig markierten Beschreibungen; das muss in der Auswertung benannt
  werden.
- Das Manifest wird nicht automatisch aktualisiert. Ändert sich das
  Referenzmanifest oder ein Bild, stoppt die Evaluation und verlangt eine
  manuelle Prüfung.

## Offene Zuordnung der Bezeichnungen 1, 2a, 2b und 3

Die vorhandenen Dateien definieren bislang ausschließlich die Varianten A bis
D. Eine belastbare ursprüngliche Zuordnung der vorgesehenen Bedingungen ohne,
mit fünf beziehungsweise mit zwanzig Referenzbildern sowie mit zusätzlichen
fachlichen Informationen zu den Bezeichnungen 1, 2a, 2b und 3 wurde im Projekt
nicht gefunden. Diese Zuordnung bleibt deshalb bewusst offen.

Der technisch vorbereitete Modus `--evaluation-variant 1` greift nicht auf das
Referenzmanifest oder Referenzbilder zu. Er verwendet ausschließlich die zwölf in
`Datenbank/klassifikationsregeln.json` unter `meta.active_class_codes`
dokumentierten Codes. Die Desktop-App zeigt vor jedem Lauf eine
Kontrollübersicht; Frontend, Backend und Kommandozeilenmodus leiten dieselbe
Auswahl dynamisch aus dieser zentralen Konfiguration ab.

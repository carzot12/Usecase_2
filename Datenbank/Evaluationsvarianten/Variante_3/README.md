# Variante 3 – annotierte Referenzdatenbank

Dies ist die einzige aktive Referenzquelle für Variante 3. Der Ordner
`Variante_3_alt` ist ein unverändertes Archiv und wird von der Anwendung nicht
gelesen. Die aktuelle Revision lautet
`v3-annotated-complete-2026-11-08-r4`.

Jede aktive annotierte Datei wird in `descriptions.csv` erfasst. Die dort
gespeicherten Werte `source_reference_id`, `source_relative_path` und
`source_sha256` dokumentieren die Herkunft, ohne dass Variante 3 zur Laufzeit
auf Variante 2B zugreifen muss.
`reference_snapshot.json` hält Dateihashes, Beschreibungen, Bestätigungsstatus,
Bildzahlen sowie Modell-, Prompt- und Regelwerksversion fest. Die App verändert
weder Annotationen noch Originale.

## Markierungslegende

| Markierung | Merkmale |
| --- | --- |
| Roter Kreis | KA, SA, SSA, AA |
| Blaues Rechteck | GSF, LSF, LR |
| Rotes Rechteck | sp HB, Spl HB |
| Lila Rechteck | Bu |
| Grünes Rechteck | RiSch, KB |

Auf einem Bild ist nur das jeweilige Zielmerkmal verbindlich markiert. Andere
sichtbare Merkmale müssen nicht annotiert sein.

## Wissenschaftliche Verwendung

- Modell: OpenAI `gpt-6.1-sol`
- Klassifikationsregeln: v1.4
- Prompt: `5.4-scientific-multilabel-no-ranking`
- alle bestätigten Annotationen werden bei jeder Analyse direkt übertragen
- Bilder und individuelle Beschreibungen werden ausschließlich aus `Variante_3` geladen
- 2B-Daten dienen nur als eingefrorener Herkunftsnachweis im eigenen CSV/Manifest
- kein KI-generiertes Referenzgedächtnis und keine automatische Reduktion
- keine Ground Truth und keine früheren Ergebnisse im Modellkontext

Der vollständige Sollbestand von 104 Bildern ist vorhanden und eindeutig
zugeordnet. Es gibt keine zusätzliche oder nicht zugeordnete Bilddatei.

Quelle der individuellen Texte ist die Pocket-Aufzeichnung
„Annotation Holzschäden Zugproben Bildanalyse“ vom 8. November 2026
(`d524e581-910f-4bdc-bcd2-ee6eea7eb929`). 79 Aussagen konnten eindeutig
zugeordnet werden. 24 weitere individuelle Beschreibungen wurden anschließend
direkt fachlich vorgegeben. `SA/DSC01872` besitzt bewusst keine individuelle
Bildbeschreibung und verwendet ausschließlich die allgemeine SA-Definition aus
Regelwerk v1.4. Damit besitzen 103 Bilder individuelle Beschreibungen und ein
Bild einen dokumentierten Regelwerksfallback.

Alle 104 aktuellen annotierten Dateifassungen und Beschreibungen wurden von CZ
fachlich bestätigt. Variante 3 ist für die Planung eines neuen, getrennten
OpenAI-Testlaufs mit `gpt-6.1-sol` **BEREIT**. Ein Testlauf wird durch diese
Metadatenaktualisierung nicht automatisch gestartet.


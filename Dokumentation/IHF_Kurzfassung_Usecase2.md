# KI-gestützte visuelle Dokumentation von Versagensmerkmalen an Holz-Zugprüfkörpern mittels regelbasiertem Prompt Engineering

*Kurzfassung – Arbeitsgrundlage für den IHF-Beitrag 2026 · Stand 04.11.2026*

## Zielsetzung

Nach Zugversuchen parallel zur Faserrichtung werden die sichtbaren Holzmerkmale und das Bruchbild zerstörter Prüfkörper bislang manuell anhand von Fotografien dokumentiert. Im Rahmen der Masterarbeit „Automatisierung von Arbeitsprozessen in der Holzbauforschung unter Anwendung von KI-gestütztem Prompt Engineering“ (TU Graz) wird untersucht, ob multimodale Sprachmodelle diese Sichtung unterstützen können. Die KI soll sichtbare Merkmale beschreiben und Klassen vorschlagen. Die Bestimmung der Versagensursache bleibt ausdrücklich der Fachperson vorbehalten.

## Methodik

Grundlage ist ein Klassifikationssystem mit zwölf gleichwertigen sichtbaren Merkmalen: GSF, LSF, KA, SA, sp HB, Spl HB, RiSch, LR, Bu, AA, SSA und KB. Alle zwölf besitzen vollständige Regeln der Version 1.4 und bleiben entsprechend ihrem dokumentierten Prüfstatus kenntlich. KB wird neutral als sichtbare Prüf- beziehungsweise Einspannspur auf einer sichtbaren Brettfläche und nicht als Holzmerkmal behandelt. Die Regeln trennen sichtbares Holzmerkmal, Bruchbild, Prüfspur und Versagensursache. Sie berücksichtigen die tatsächliche Aufnahmesituation: Draufsichten beider Breitseiten der auf dem Boden liegenden Bruchstücke, keine systematischen Stirnholz- oder Schmalseitenansichten.

Das Prompt Engineering ist regelbasiert, schema-gebunden und zweiphasig angelegt. Ein Systemprompt begrenzt die Aufgabe auf sichtbare Merkmale. Der Benutzerprompt verlangt zuerst neutrale Beobachtungen und erst danach unverbindliche Klassenkandidaten. Jeder Kandidat wird unabhängig von seiner unveränderten Konfidenz als `sicher_sichtbar` oder `vermutet` gekennzeichnet. Der Prompt erlaubt vollständige Mehrfachklassifikation, verlangt bei unzureichender Evidenz „UNKLAR“, verbietet erfundene Schwellenwerte und erzwingt keine Bruchbildklasse. Eine optionale Primärempfehlung ist keine nachgewiesene Versagensursache. Die Klassenregeln werden als strukturiertes JSON übertragen. Die Ausgabe ist durch ein JSON-Schema mit festen Klassencodes vorgegeben und wird anschließend im Code erneut geprüft, unter anderem durch einen Filter, der kausale Ursachenbehauptungen zurückweist. Informationen, die das Ergebnis vorwegnehmen könnten, werden nicht übertragen: Versuchsprotokolle, Ground Truth, frühere Ergebnisse, Expertenbewertungen, Dateinamen und Prüfkörpernummern.

## Technische Umsetzung

Die Anwendung ist eine native macOS-Desktop-Anwendung (React-Oberfläche in pywebview, Python-Backend). Angebunden sind ein lokales Modell, Qwen3-VL 4B über die Laufzeitumgebung Ollama, sowie die Cloud-Modelle `gpt-6.1-sol` (OpenAI) und `ministral-3b-2512` (Mistral). Alle drei erhalten denselben fachlichen Prompt. Nummernfotos werden automatisch von der Analyse ausgeschlossen. Die Analysebilder werden vorausgewählt und können manuell angepasst werden. Für die Übertragung werden die Originale nur gelesen und im Arbeitsspeicher verkleinert. Jeder Lauf wird mit eindeutiger Kennung, Modell, Laufzeit, Prompt- und Regelversion sowie SHA-256-Prüfsummen von Prompt, Bildern und Referenzen gespeichert. Die fachliche Bewertung (Bestätigung, Teilkorrektur, Ablehnung, fachliche Hauptklasse, weitere Merkmale, Begründung) wird getrennt von der unveränderten KI-Ausgabe abgelegt und gemeinsam mit ihr als CSV und JSON exportiert. Cloud-Aufrufe erfordern eine ausdrückliche Kosten- und Übertragungsfreigabe. API-Schlüssel werden ausschließlich aus dem macOS-Schlüsselbund bzw. der Umgebung gelesen.

## Evaluationsdesign

Implementiert ist eine referenzfreie Variante 1: Das Modell erhält ausschließlich die ausgewählten Prüfkörperbilder und die Beschreibungen der zwölf Merkmale. Das Referenzmanifest wird in diesem Modus technisch nicht geöffnet, und vor jedem Lauf wird eine Kontrollübersicht erzeugt. Die Varianten 2, 2a und 3 verwenden ihre getrennt manifestierten Referenzbestände; Variante 3 ergänzt die Beschreibungen zum Bildbestand von 2a. Die fachliche Abdeckung der Ground Truth bleibt von der aktiven Merkmalsliste getrennt.

## Tatsächlicher Entwicklungsstand

Die Anwendung ist funktionsfähig, als App-Bundle installiert und durch 65 automatisierte Tests mit simulierten Modellantworten abgesichert (alle bestanden, Stand 26.10.2026). Mit früheren Entwicklungsständen (29 zulässige Klassen, ältere Prompts) wurden einzelne Läufe aller drei Modelle an Prüfkörpern einer Testserie technisch erfolgreich durchgeführt. Mit dem aktuellen Klassen- und Promptstand liegt noch kein Modelllauf vor, und die Evaluation ist noch nicht ausgeführt. **Aussagen über die Klassifikationsgüte sind daher derzeit nicht möglich.**

## Grenzen und Ausblick

Die verfügbaren Draufsichten lassen Faserneigung in Dickenrichtung, Astverläufe auf der Schmalseite und Jahrringbindungen von Trennflächen häufig nicht beurteilen. Mehrere Klassenregeln sind fachlich noch als offen oder nur bedingt geeignet eingestuft; für Grenzen wie global/lokal oder spröde/splitternd fehlen kalibrierte Kriterien. Die Zahl bestätigter Referenzbilder ist gering. Generative Modelle liefern zudem auch bei identischen Eingaben nicht zwingend identische Antworten, und vom Modell angegebene Konfidenzen sind nicht kalibriert. Nächste Schritte sind die Festlegung der Varianten 2a bis 3, ein vom Referenzpool unabhängiger, fachlich annotierter Testdatensatz, wiederholte Läufe zur Stabilitätsbewertung, Multi-Label-Metriken sowie standardisierte Aufnahmen. Validierte Ergebnisse könnten über die vorhandenen Exporte an die eigenständige Anwendung für statistische Auswertung und Prüfberichtserstellung (Use Case 3) übergeben werden.

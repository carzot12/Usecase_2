# Offene Punkte und Validierungsbedarf – KI-Bildanalyse (Use Case 2)

**Stand:** 26.10.2026 · Grundlage: Code- und Datenanalyse, siehe [Technischer_Bericht_KI_Bildanalyse.md](Technischer_Bericht_KI_Bildanalyse.md)

Priorität: **H** = vor einer wissenschaftlichen Auswertung oder dem IHF-Beitrag zu klären · **M** = für Weiterentwicklung oder Übergabe wichtig · **N** = Verbesserung.

---

## A. Fachliche Punkte (Holzbau / Klassifikation)

| Nr. | Prio | Offener Punkt | Quelle |
|---|---|---|---|
| A1 | H | **Fachlicher Prüfstatus der Regeln:** Nur KA ist „vorläufig bestätigt“; GSF, LSF, SA, RiSch sind „bedingt geeignet“, sp HB, Spl HB, LR „fachlich offen“. Eine fachliche Freigabe (z. B. durch Betreuung oder Laborleitung) ist nicht dokumentiert. | `klassifikationsregeln.json` → `fachlicher_pruefstatus` |
| A2 | H | **GSF/LSF:** Bezugs- und Mindestlänge für „global“, räumliche Ausdehnung der lokalen Zone, zulässige Winkelstreuung, Umgang mit LSF ohne sichtbares Störungszentrum. | Regeln GSF, LSF |
| A3 | H | **KA/SA:** Abgrenzung zu Schmalseitenästen (SSA) für die verwendeten Querschnitte, Behandlung minimaler Kantenanschnitte, Astlöcher und herausgebrochene Äste, durchlaufende Äste; Koexistenzregeln mit LSF. | Regeln KA, SA |
| A4 | H | **sp HB / Spl HB:** messbare Grenze (Bruchlänge, Splitterlänge, Faserauszug), Regel für gemischte Brüche, Beibehaltung des Begriffs „spröde“ ohne Kraft-Verformungs-Daten. | Regeln sp HB, Spl HB |
| A5 | H | **LR:** Festlegung, ob LR als Holzmerkmal, Bruchmerkmal oder kontextabhängig geführt wird; Abgrenzung eines einzelnen LR von mehreren Längsaufspaltungen bei Spl HB. | Regel LR |
| A6 | H | **RiSch:** Laborregel für die Vergabe bei schichtartiger Ablösung ohne nachweisbare Jahrringgrenze; Kalibrierung gemischter RiSch/Spl-HB-Morphologien. | Regel RiSch |
| A7 | H | **Begriff „Hauptversagensursache“ im Review:** Definition, nach welchen Kriterien die Fachperson die Hauptklasse vergibt (sichtbares Merkmal vs. maßgebende Ursache), und wie mit Fällen ohne eindeutiges Primärversagen in der Auswertung umgegangen wird. | Expertenreview |
| A8 | M | Übertragbarkeit der Regeln auf andere Holzarten, Feuchten und Probengeometrien. | Regeln, `gueltigkeitsbereich` |
| A9 | M | Aufnahme weiterer Klassen aus den 21 nicht aktiven Definitionen (19 von 29 mit `review_required=true`). | `versagensarten_erweitert.json` |
| A10 | M | Literatur- und Normquellen: Laut Regeldatei wurden mehrere Normvolltexte (DIN 4074-1, ASTM D143) nicht eingesehen; teils nur Abstract (Hunger und van de Kuilen 2018). Für eine Veröffentlichung sind die Quellen selbst zu prüfen. | `meta.quellenpruefung` |
| A11 | M | Fehlende Aufnahmeperspektiven: Entscheidung, ob künftig Bruchflächen-, Schmalseiten- oder Stirnholzaufnahmen standardisiert ergänzt werden. | Fotodokumentation |

## B. Wissenschaftliche Punkte (Evaluation)

| Nr. | Prio | Offener Punkt |
|---|---|---|
| B1 | H | **Definition der Varianten 2a, 2b und 3** (Art und Anzahl der Referenzbilder, zusätzliche fachliche Informationen, Auswahlverfahren). Im Projekt ist keine verbindliche Zuordnung vorhanden; der Code lehnt diese Varianten ab. |
| B2 | H | **Verhältnis der Varianten 1–3 zu den implementierten Varianten A–D** des Evaluationswerkzeugs. A–D unterscheiden sich teils in mehreren Faktoren gleichzeitig (B nutzt erweiterte Beschreibungen v1.0, D Stufe 2 die Regeln v1.2). |
| B3 | H | **Unabhängiger Testdatensatz:** Die vorhandene Ground Truth besteht aus den Referenzbildern selbst (interne Leave-one-Prüfkörper-out-Evaluation). Für eine belastbare Aussage wird ein fachlich annotierter, vom Referenzpool getrennter Datensatz benötigt (Prüfkörper, Bilder, Klassen, annotierende Person, Datum). |
| B4 | H | **Fallzahlen:** 18 bestätigte Fälle für die aktiven Klassen (1–3 je Klasse), LR ohne Fall. Erforderliche Mindestzahl je Klasse ist festzulegen. |
| B5 | H | **Metriken für Mehrfachklassifikation:** Implementiert sind nur Metriken der Primärklasse. Festzulegen ist, wie sekundäre Merkmale, „UNKLAR“ und „Kein eindeutiges Primärversagen“ bewertet werden. |
| B6 | H | **Wiederholungsläufe:** Anzahl der Läufe je Bedingung zur Bewertung der Ausgabestabilität. |
| B7 | H | **Vergleichbarkeit der Modelle:** unterschiedliche Bildauflösung (768 px lokal, 2 048 px Cloud), fehlende Temperaturfestlegung bei OpenAI, Mistral-Teilanfragen mit je einem Testbild im Standardmodus. Entscheidung, ob angeglichen oder als Bestandteil der Konfiguration berichtet wird. |
| B8 | M | **Interrater-Vergleich:** Ob eine zweite Fachperson Bewertungen vornimmt, um die Unsicherheit der Ground Truth abzuschätzen. |
| B9 | M | **Umgang mit historischen Läufen:** 19 App-Läufe und die CLI-Testläufe beruhen auf 29 Klassen und älteren Prompts. Festzulegen, ob sie nur als Entwicklungsdokumentation dienen (empfohlen) oder ausgewertet werden. |
| B10 | M | **Kostenrahmen und Freigabe** für Cloud-Läufe (Anzahl Anfragen je Variante; die Evaluation A–D umfasst 90 Aufrufe je Modell). |
| B11 | M | **Verzerrungen durch Prompt:** Einfluss von Klassenreihenfolge, Beispielkombinationen und `confirmed_reference_count` auf die Vorschläge ist nicht untersucht. |

## C. Technische Punkte

| Nr. | Prio | Offener Punkt | Ort |
|---|---|---|---|
| C1 | H | **Codestand versionieren:** Regeln 1.2, Variante 1, Referenzspeicher und Tests sind nicht committet. Vor Evaluationsläufen committen (Prompt-Hash hängt vom Code ab). | Git |
| C2 | H | **Leakage-Prüfung in Variante 1:** Ohne geöffnetes Manifest werden Testbilder nicht gegen bekannte Referenz-/Kontextbilder abgeglichen. Für Evaluationsdatensätze eine manifestunabhängige Hashprüfung ergänzen oder organisatorisch sicherstellen. | `load_catalog(include_references=False)` |
| C3 | H | **Widerspruch Mistral-Live-Status:** `Tool/README.md` meldet keinen Live-Aufruf seit der 422-Korrektur; Ergebnisdateien mit `mode: live` vom 23./24.10.2026 existieren. Klären und README korrigieren. | `Tool/README.md`, `Output/…/Ergebnisse/` |
| C4 | H | **Rohantworten speichern:** Validierte Ergebnisse werden gespeichert, Rohantworten und fehlgeschlagene Antworten nicht. Für die wissenschaftliche Fehleranalyse (ungültige JSON-Antworten, Kausalfilter-Treffer) erforderlich. | `run_analysis` |
| C5 | M | **Veraltete Texte:** Banner „Sieben dokumentierte Hauptversagensklassen“; Fehlermeldung „eine der 29 Klassen“; `Tool/README.md` (Mock als Standard). | `App.tsx:102`, `ki_bildanalyse.py:978` |
| C6 | M | **Expertenreview:** keine Versionshistorie, keine Erfassung der bewertenden Person; Server akzeptiert 29 Codes, Oberfläche 8. | `bridge.py:1007-1038` |
| C7 | M | **Export:** Spalte „Fehlermeldung“ immer leer; Prompt-Hash und Bild-Hashes fehlen in der CSV; fehlgeschlagene Läufe nicht exportierbar. | `export_selected_runs` |
| C8 | M | **App-Referenzspeicher:** nicht an Modellaufrufe angebunden, Manifest noch nicht angelegt (28 unregistrierte Bilder); Klassenänderung verschiebt Datei nicht; Bestätigung ohne Personenkennung. Entscheidung, ob er den kuratierten Bestand ablösen soll. | `reference_store.py` |
| C9 | M | **Batches ohne Evaluationsvariante:** Serienläufe nur im Standardmodus; Variante 1 nur einzeln. | `_run_batch` |
| C10 | M | **Kausalitätsfilter** erweitern (Englisch, Umschreibungen); Längenbegrenzungen des Schemas im Code prüfen. | `CAUSAL_CLAIM` |
| C11 | M | **OpenAI-Parameter:** Temperatur und Timeout nicht gesetzt. | `call_openai_with_metadata` |
| C12 | M | **Duplikatprüfung beim Bildimport** für Prüfkörperbilder fehlt. | `import_images` |
| C13 | M | **Prompt-Hash** umfasst weder Provider/Parameter noch die übertragenen Bildbytes; Promptversion wird manuell gepflegt. Dokumentieren oder erweitern. | `run_analysis` |
| C14 | M | **Reproduzierbarkeit der Umgebung:** `package.json` mit `latest`; Python nur Mindestversionen; Bundle mit CPython 3.13, Entwicklung mit 3.12. Versionen fixieren. | `package.json`, `requirements*.txt` |
| C15 | M | **Auslieferung:** nur Ad-hoc-Signatur, keine Notarisierung, kein Installationsskript; zahlreiche Bundle-Duplikate in `dist/`. | `build_macos.sh` |
| C16 | M | **Regeldateien im Bundle:** Änderungen an `klassifikationsregeln.json` wirken in der installierten App erst nach Neubau. Bei jeder Regeländerung neu bauen und Hash prüfen. | `KI-Bildanalyse.spec` |
| C17 | N | Ergebnis-Zuordnung über Dateinamenpräfix (Kollision bei Prüfkörpernamen, die Präfix anderer sind). | `_result_records` |
| C18 | N | Sortierwerkzeug: Serienkennung `Z-30` fest codiert; Abhängigkeit von einem Skript unter `Datenbank/Alte_Versuche/`. | `bildzuordnung_neue_analysen.py` |
| C19 | N | Leere temporäre Testordner unter `Pruefdaten/Neue_Analysen/` (sieben `tmp…`). | Tests |
| C20 | N | Frontend-Tests und End-to-End-Tests fehlen. | `frontend/` |
| C21 | N | Variante D: Stufe 1 erzwingt `UNKLAR` im gemeinsamen Schema; eigenes Schema für Merkmalsextraktion wäre sauberer. In Variante C/Mistral wird die ungekürzte Referenzanzahl protokolliert. | `evaluation.py` |

## D. Datenschutz und Organisation

| Nr. | Prio | Offener Punkt |
|---|---|---|
| D1 | H | Freigabe der Forschungseinrichtung für die Übertragung von Versuchsfotos an OpenAI und Mistral; Prüfung der Speicher- und Trainingsbedingungen der Anbieter (für Mistral wird kein Speicherverbot gesetzt). |
| D2 | M | Gemeinsame Nutzung des OpenAI-Schlüsselbunddienstes „KI-Bemessungstool OpenAI“ mit Use Case 1: Kostenzuordnung und Zugriffsrechte klären. |
| D3 | M | Archivierung: Ergebnis-, Bewertungs- und Exportdateien sind nicht schreibgeschützt und nicht signiert; absolute Pfade mit Benutzernamen in den Dateien. |
| D4 | M | Übergabe: Verantwortliche Person für Schlüssel, Ollama-Installation und Modellversionen festlegen. |

## E. Für die wissenschaftliche Evaluation und den IHF-Beitrag noch fehlende Informationen

1. Verbindliche Definition der Varianten 2a, 2b und 3 (B1, B2).
2. Ein fachlich annotierter, vom Referenzpool unabhängiger Testdatensatz mit dokumentierter Annotation (B3, B4).
3. Fachliche Freigabe bzw. Überarbeitung der acht Regeln, insbesondere der „fachlich offenen“ Klassen (A1–A6).
4. Festgelegte Bewertungsmetriken für Mehrfachklassifikation und „UNKLAR“ (B5).
5. Durchgeführte Läufe mit dem aktuellen Stand (Regeln 1.2, Prompt 3.1) – derzeit keine (B9).
6. Anzahl der Wiederholungen und Kostenfreigabe für Cloud-Modelle (B6, B10, D1).
7. Versionierter Codestand mit Commit-Kennung für die Methodenbeschreibung (C1).
8. Genaue Modellversionen zum Zeitpunkt der Läufe (lokaler Ollama-Modell-Digest, von den Cloud-Anbietern gemeldete Modellkennungen).
9. Beschreibung der Prüfkörper und Versuche (Holzart, Festigkeitsklasse, Querschnitt, Feuchte, Prüfmaschine, Serie Z-30), die nicht Teil dieser Software sind.
10. Geprüfte Literaturquellen für die Methodik (A10).

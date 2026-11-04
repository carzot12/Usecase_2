# Technischer Bericht: KI-gestützte Bildanalyse von Holz-Zugprüfkörpern (Use Case 2)

**Masterarbeit:** „Automatisierung von Arbeitsprozessen in der Holzbauforschung unter Anwendung von KI-gestütztem Prompt Engineering“, TU Graz
**Gegenstand:** Desktop-Anwendung „KI-Bildanalyse“, Version 1.0.0 (Bundle-ID `at.holzbauforschung.ki-bildanalyse`)
**Stand der Analyse:** 26.10.2026
**Grundlage:** Quellcode, Konfigurations- und Datendateien im Projektverzeichnis `Usecase_2/`, vorhandene Ergebnisdateien sowie ein aktueller lokaler Testlauf (siehe Kapitel 11). Es wurden keine Modelle gestartet und keine API-Aufrufe ausgeführt.

> **Aktualisierung 03.11.2026:** Die aktive Merkmalsliste umfasst nun zwölf
> gleichwertige Codes und wird aus
> `Datenbank/klassifikationsregeln.json/meta.active_class_codes` geladen.
> KB ist dabei als sichtbare Prüf- beziehungsweise Einspannspur und nicht als
> Holzmerkmal definiert. Das Regelwerk liegt in Version 1.3 vor.
> Promptversion `5.2-qwen-joint-images-primary-candidate` wertet alle Ansichten
> gemeinsam aus und unterscheidet fachliches `UNKLAR`, `KEINE_MERKMALE` und
> technische Antwortfehler. Ein lokaler, referenzfreier Qwen-Test mit vier
> Bildern von Z-30-01 endete mit `done_reason=stop`, gültigem Schema und den
> visuellen Vorschlägen Spl HB sowie LR. Historische Abschnitte dieses Berichts
> dokumentieren teilweise den früheren Acht-Klassen-Stand.

---

## Inhaltsverzeichnis

1. Einleitung und Zielsetzung
2. Methodik der Dokumentation und Kennzeichnung des Nachweisstands
3. Projektstruktur
4. Systemarchitektur
5. Bilddaten und Bildworkflow
6. Klassifikationssystem
7. Prompt Engineering
8. KI-Modelle und Modellanbindung
9. Evaluationskonzept
10. Referenzbilddatenbank
11. Ergebnisse, Expertenreview und Export
12. Qualitätssicherung, Tests, Build und Installation
13. Datenschutz und Sicherheit
14. Diskussion und Grenzen
15. Ausblick
16. Zusammenfassung des Entwicklungsstands
Anhang A: Exemplarischer Prompt (ohne Bilddaten, ohne Modellaufruf)
Anhang B: Ergebnisschema 2.0
Anhang C: Verzeichnis der zitierten Projektdateien

---

## 1. Einleitung und Zielsetzung

In Zugversuchen an Holz parallel zur Faserrichtung wird nach dem Versagen dokumentiert, welche Holzmerkmale und welches Bruchbild am zerstörten Prüfkörper sichtbar sind und welche Versagensursache fachlich als maßgebend angesehen wird. Diese Dokumentation erfolgt bislang manuell anhand von Fotografien. Use Case 2 untersucht, inwieweit multimodale Sprachmodelle (Vision-Language-Modelle) diese Sichtung unterstützen können, wenn ihr Verhalten durch strukturiertes Prompt Engineering, fachlich formulierte Klassifikationsregeln und technische Schutzmechanismen begrenzt wird.

Die implementierte Anwendung verfolgt dabei ein bewusst eingeschränktes Ziel:

- Das Modell **beschreibt sichtbare Merkmale** und **schlägt unverbindliche Klassenkandidaten** aus einer festgelegten Klassenliste vor.
- Das Modell **stellt keine Versagensursache fest**. Kausale Formulierungen werden technisch zurückgewiesen (Abschnitt 7.6).
- Die fachliche Hauptversagensursache bestimmt die Anwenderin im **Expertenreview**; diese Bewertung wird getrennt von der KI-Ausgabe gespeichert (Kapitel 11).

Use Case 2 ist eigenständig. Es besteht keine Datenverbindung zum Bemessungswerkzeug aus Use Case 1. Die Anwendung übernimmt lediglich dessen Architekturmuster (React-Oberfläche in einer pywebview-Desktop-Hülle) und den Namen des macOS-Schlüsselbunddienstes für den OpenAI-Schlüssel (Abschnitt 13.3). Use Case 3 (statistische Auswertung, Prüfberichte) ist eine nachgelagerte, eigenständige Anwendung und nicht Teil dieser Software.

## 2. Methodik der Dokumentation und Kennzeichnung des Nachweisstands

Der Bericht dokumentiert ausschließlich die tatsächlich implementierte Software. Alle relevanten Python-Module, das Frontend, die Klassifikationsdateien, die Referenzmanifeste, die Testdateien sowie die vorhandenen Ergebnis-, Bewertungs- und Batchdateien wurden gelesen. Zusätzlich wurden folgende lokale, nicht destruktive Prüfungen ausgeführt:

- vollständiger Lauf der automatisierten Tests (`python3 -m unittest discover -s tests`),
- TypeScript-Typprüfung des Frontends ohne Ausgabe (`tsc -p tsconfig.app.json --noEmit`),
- Signaturprüfung der vorhandenen App-Bundles mit `codesign --verify`,
- Abgleich der SHA-256-Prüfsummen von Regeldateien und Referenzmanifest,
- Erzeugung des Prompts aus dem Code (ohne Bilder, ohne Modellaufruf) und Vergleich mit der Beispieldatei.

Der Nachweisstand wird im Bericht wie folgt gekennzeichnet:

| Kennzeichnung | Bedeutung |
|---|---|
| **[I+T]** | implementiert und durch automatisierte Tests abgedeckt (Mock- oder Fake-Transport, kein echter Modellaufruf) |
| **[I+P]** | implementiert und durch vorhandene Ergebnisdateien praktisch nachweisbar ausgeführt (historische Läufe) |
| **[I]** | implementiert, aber weder durch Tests noch durch vorhandene Läufe praktisch validiert |
| **[G]** | geplant bzw. fachlich beschlossen, nicht implementiert |
| **[E]** | bekannte Einschränkung |

„Praktisch validiert“ bedeutet hier nur, dass eine Funktion technisch ausgeführt wurde. Eine **fachliche oder statistische Validierung der Klassifikationsgüte liegt nicht vor** (Kapitel 9 und 14).

## 3. Projektstruktur

| Pfad | Inhalt |
|---|---|
| `app.py` | Einstiegspunkt; ruft `desktop.main.main()` auf |
| `desktop/main.py` | pywebview-Fenster, Splash-Bildschirm, Ressourcenauflösung im Bundle |
| `desktop/bridge.py` | `DesktopBridge`: Schnittstelle zwischen React und Python |
| `desktop/reference_store.py` | Verwaltung des App-Referenzspeichers `Datenbank/Referenzbilder_neu/` |
| `Tool/ki_bildanalyse.py` | Klassifikationskern und Kommandozeilenwerkzeug |
| `Tool/evaluation.py` | Evaluationswerkzeug (Varianten A–D, Kommandozeile) |
| `Tool/bildzuordnung_neue_analysen.py` | OCR-gestützte Sortierung neuer Fotoserien (Kommandozeile) |
| `Tool/openai_config.py`, `Tool/mistral_config.py` | Schlüsselauflösung und sichere Fehlermeldungen |
| `Tool/model_verification.json`, `Tool/mistral_model_verification.json` | lokal dokumentierte Modellfähigkeiten |
| `Tool/mock_response.json` | feste Mock-Antwort (`UNKLAR`) |
| `Datenbank/klassifikationsregeln.json` | aktive Klassifikationsregeln, Version 1.2, acht Klassen |
| `Datenbank/Archiv/klassifikationsregeln_v1.1.json` | archivierte Vorversion (sieben Klassen) |
| `Datenbank/versagensarten.json` | 29 Codes mit Kurzbezeichnung |
| `Datenbank/versagensarten_erweitert.json` | 29 erweiterte Klassenbeschreibungen, Version 1.0 |
| `Datenbank/klassifikationsprompt_v1.3_beispiel.json` | vollständiger Beispielprompt (Variante 1) |
| `Datenbank/Pruefung_Versagensarten.md` | fachliche Prüfung der Klassendefinitionen (Stand 21.10.2026) |
| `Datenbank/Referenzbilder/` | kuratierte Referenzbilder mit `referenzbilder_manifest.json` (Schema 2.0) |
| `Datenbank/Referenzbilder_neu/` | App-Referenzspeicher mit `aktive_hauptversagensarten.json` |
| `Evaluation/` | `ground_truth_manifest.json`, `README.md` |
| `Pruefdaten/Neue_Analysen/<Projekt>/Sortiert/<PK>/` | sortierte Prüfkörperbilder |
| `Output/EKI_Bildanalyse/Projekte/<Projekt>/` | Projektdaten der App |
| `Output/KI_Klassifikation/`, `Output/KI_Evaluation/` | Ergebnisse der Kommandozeilenwerkzeuge |
| `frontend/` | React/TypeScript-Quellen und Build-Ausgabe `dist/` |
| `tests/` | vier Testmodule mit 65 Testfällen |
| `KI-Bildanalyse.spec`, `build_macos.sh` | PyInstaller-Spezifikation und Build-Skript |
| `requirements-app.txt`, `requirements-build.txt`, `Tool/requirements.txt` | Python-Abhängigkeiten |

Weitere Ordner (`backups/`, `tmp/`, `xxx/`, `allgemein/`, `build/`, `dist/`) enthalten Sicherungen, Zwischenstände, Literatur-PDFs und Build-Artefakte; sie sind nicht Teil der Laufzeitlogik und laut `.gitignore` von der Versionsverwaltung ausgeschlossen.

## 4. Systemarchitektur

Die ausführliche Architekturbeschreibung mit Diagrammen befindet sich in [Systemarchitektur_KI_Bildanalyse.md](Systemarchitektur_KI_Bildanalyse.md). Hier werden die für die wissenschaftliche Bewertung wesentlichen Eigenschaften zusammengefasst.

### 4.1 Aufbau

Die Anwendung besteht aus einer React-Oberfläche, die in einem nativen WebKit-Fenster (pywebview) läuft, und einem Python-Prozess, der alle Datei-, Analyse- und Netzwerkoperationen ausführt. Das Frontend hat keinen direkten Dateisystem- oder Netzwerkzugriff; es ruft ausschließlich die Methoden der Klasse `DesktopBridge` auf ([desktop/bridge.py:59](../desktop/bridge.py#L59)). **[I+T]**

### 4.2 Ablauf einer Analyseanfrage

1. Die Anwenderin wählt einen Prüfkörper; `specimen_detail` liefert alle Bilder als verkleinerte Vorschau (720 px, JPEG-Qualität 84) sowie die gespeicherte Bildauswahl und die Analysehistorie.
2. Jede Änderung der Auswahl wird über `save_image_selection` sofort in `Bildauswahl/<PK>.json` gespeichert. Das Frontend wartet vor jedem Start auf den Abschluss dieser Speichervorgänge.
3. `start_analysis` prüft Prüfkörper, Bildauswahl, Provider/Modell-Kombination, bei Cloud-Modellen die Kostenbestätigung und bei Variante 1 das Kontrollübersichts-Token. Ein identischer, bereits laufender Auftrag (gleicher Prüfkörper, Provider und Modell) wird nicht dupliziert ([desktop/bridge.py:661-696](../desktop/bridge.py#L661-L696)). **[I+T]**
4. Ein Hintergrund-Thread ruft `run_analysis` auf. Fortschrittsmeldungen („Modell wird geladen“, „Bilder werden analysiert“, „Ergebnis wird verarbeitet“) werden im Job-Status abgelegt und vom Frontend angezeigt.
5. `run_analysis` lädt Klassen und Regeln, prüft Bilder, wählt gegebenenfalls Referenzen, baut den Prompt, ruft den Provider auf, validiert die Antwort und schreibt die Ergebnisdatei ([Tool/ki_bildanalyse.py:1198-1454](../Tool/ki_bildanalyse.py#L1198-L1454)).
6. Nach Abschluss lädt das Frontend Projekt, Detailansicht und Ergebnisliste neu.

### 4.3 Projektverwaltung

Ein „Projekt“ ist ein Ordner mit Prüfkörper-Unterordnern. Beim Öffnen wird automatisch ein Unterordner `Sortiert` verwendet, falls vorhanden ([desktop/bridge.py:174-183](../desktop/bridge.py#L174-L183)). Neue Projekte werden unter `Pruefdaten/Neue_Analysen/<Name>/Sortiert` angelegt. Die Projekt-Metadaten (`project.json`) und alle App-Daten liegen getrennt von den Bildern unter `Output/EKI_Bildanalyse/Projekte/<Projekt>/`. Namen werden über `_safe_name` auf Buchstaben, Ziffern, Umlaute, `_`, `.` und `-` beschränkt. **[I+T]**

### 4.4 Modell- und Providerkonfiguration

Die zulässigen Kombinationen sind fest in `MODEL_CONFIGS` hinterlegt; andere Kombinationen – darunter das frühere LLaVA – werden abgewiesen ([desktop/bridge.py:44-48](../desktop/bridge.py#L44-L48); Test `test_removed_llava_choice_cannot_start_a_new_analysis`). **[I+T]** Für Ollama werden daraus Kontextgröße (24 576 Token) und Generationsbudget (4 096 Token) übernommen; die für OpenAI und Mistral hinterlegten Zahlenwerte werden im Code nicht ausgewertet. In der Kommandozeile kann die OpenAI-Modellkennung über `USECASE2_OPENAI_MODEL` und die Ollama-Kennung über `USECASE2_OLLAMA_MODEL` überschrieben werden; die Oberfläche verwendet ausschließlich die festen Kennungen.

### 4.5 Fehlerbehandlung

Fehler werden als `AnalysisError` beziehungsweise providerspezifische Fehler ausgelöst, im Worker-Thread abgefangen und als lesbare Meldung am Auftrag angezeigt. Fehlermeldungen von OpenAI und Mistral enthalten Typ, Modell, HTTP-Status und Request-ID, aber keine Schlüssel oder Bildinhalte ([Tool/openai_config.py](../Tool/openai_config.py), [Tool/mistral_config.py](../Tool/mistral_config.py)). **[I+T]** Eine automatische Wiederholung fehlgeschlagener Aufrufe ist nicht implementiert. **[E]**

## 5. Bilddaten und Bildworkflow

### 5.1 Versuchs- und Aufnahmebedingungen

Die Bilder stammen aus Zugversuchen an Holz parallel zur Faserrichtung. Nach dem Versuch werden die Bruchstücke auf den Boden gelegt und von oben fotografiert; anschließend werden sie umgedreht und von der gegenüberliegenden Breitseite fotografiert. Systematische Stirnholz-, Querschnitts- oder Schmalseitenaufnahmen liegen nicht vor. Diese Rahmenbedingungen sind auch in den aktiven Klassifikationsregeln festgehalten (`meta.fotodokumentation` in [Datenbank/klassifikationsregeln.json](../Datenbank/klassifikationsregeln.json)) und werden damit indirekt Teil des Prompts, weil die Regeln einzelner Klassen auf fehlende Ansichten verweisen.

Vor jedem Prüfkörper wird ein Identifikationsfoto mit einer Nummerntafel aufgenommen („Nummernfoto“).

### 5.2 Sortierung neuer Serien (Kommandozeile)

Das Werkzeug [Tool/bildzuordnung_neue_analysen.py](../Tool/bildzuordnung_neue_analysen.py) ordnet unsortierte Fotos Prüfkörpern zu. **[I+P]** (Ergebnis liegt für `Test_001` vor: `Sortiert/Bildzuordnung.csv`, `Sortierprotokoll.json`.)

- Es liest nur direkt im Eingabeordner liegende Bilder, verlangt EXIF-Aufnahmezeiten und bricht ab, wenn EXIF-Zeit fehlt, doppelt ist oder der Dateinamenreihenfolge widerspricht.
- Lokale OCR (aus dem Altskript `Datenbank/Alte_Versuche/Zugversuch 2/bildzuordnung_serien.py`) erkennt Kennungen der Form `Z-30-NN`; die Serienkennung `30` ist im regulären Ausdruck fest codiert. **[E]**
- Zufällig erkannte Fremdnummern werden über die längste nicht fallende Nummernfolge in Aufnahmereihenfolge herausgefiltert.
- Alle Bilder nach einem Nummernfoto werden bis zum nächsten Nummernfoto dem erkannten Prüfkörper zugeordnet. Nummernfotos erhalten beim Kopieren das Präfix `Nummer_`.
- Unklare Fälle werden nach `Sortiert/Zur_Pruefung/` kopiert; fehlende Nummern werden in `Pruefhinweise.txt` und in der CSV ausgewiesen.
- Originale werden kopiert (`shutil.copy2`), nicht verschoben. Für bis zu drei gleichmäßig verteilte Bruchbilder je Prüfkörper werden symbolische Links in `Analyseauswahl/` angelegt (Altbestand, siehe 5.4).

### 5.3 Bildimport in der App

Über „Bilder hinzufügen“ bzw. „Einzelne Bilder hochladen“ werden Dateien (`jpg`, `jpeg`, `png`, `webp`) per Dateidialog ausgewählt und in den Prüfkörperordner **kopiert** ([desktop/bridge.py:576-597](../desktop/bridge.py#L576-L597)). Bei Namensgleichheit wird ein Suffix `_2`, `_3` … angehängt; vorhandene Dateien werden nicht überschrieben. **[I]** (keine automatisierten Tests für den Dateidialog)

**[E]** Beim Import von Prüfkörperbildern findet **keine SHA-256-Duplikatprüfung** statt; dieselbe Datei kann unter anderem Namen mehrfach importiert werden. Die SHA-256-Duplikaterkennung ist nur im Referenzspeicher implementiert (Kapitel 10).

### 5.4 Nummernfotos, Analysebilder und manuelle Auswahl

- **Erkennung von Nummernfotos:** ausschließlich über das Dateinamenpräfix `nummer_` (Groß-/Kleinschreibung egal) ([desktop/bridge.py:217-219](../desktop/bridge.py#L217-L219)). Eine inhaltliche Erkennung findet in der App nicht statt. **[I+T]**
- **Behandlung:** Nummernfotos werden in der Galerie angezeigt und als „IDENTIFIKATION · NICHT ANALYSIERT“ gekennzeichnet; ihre Checkbox ist deaktiviert. `collect_test_images` schließt sie zusätzlich serverseitig aus, sodass sie nie an ein Modell übertragen werden ([Tool/ki_bildanalyse.py:294-300](../Tool/ki_bildanalyse.py#L294-L300)). **[I+T]**
- **Vorauswahl:** Beim ersten Öffnen werden bis zu drei über die Bildserie verteilte Bruchbilder vorausgewählt (Indizes `round(i·(n−1)/2)` für i = 0, 1, 2; [desktop/bridge.py:107-115](../desktop/bridge.py#L107-L115)). Existiert ein Altordner `Analyseauswahl/`, wird dessen Auswahl übernommen. **[I+T]**
- **Persistenz:** `Bildauswahl/<PK>.json` speichert `selectedImages`, `knownImages`, `selectionCustomized` und den Zeitstempel. Später hinzugekommene Bilder werden automatisch ausgewählt; ist die Auswahl nicht manuell angepasst und übersteigt sie drei Bilder, wird auf die Vorauswahl zurückgesetzt ([desktop/bridge.py:117-138](../desktop/bridge.py#L117-L138)). **[I+T]**
- **Manuelle Auswahl:** Klick auf ein Bild oder dessen Checkbox, „Alle Bruchbilder auswählen“, „Alle abwählen“. Eine Analyse ohne ausgewähltes Bild wird abgewiesen. **[I+T]**
- Die Kommandozeile verarbeitet ohne explizite Auswahl alle Bilder des Ordners bzw. des Altordners `Analyseauswahl/`.

### 5.5 Prüfung der Analysebilder und Datenintegrität

`collect_test_images` ([Tool/ki_bildanalyse.py:278-343](../Tool/ki_bildanalyse.py#L278-L343)) prüft:

1. Der Prüfkörperordner liegt innerhalb des erlaubten Wurzelverzeichnisses (in der App: Projektordner). **[I+T]**
2. Ausgewählte Dateinamen sind eindeutig, enthalten keine Pfadbestandteile und existieren. **[I+T]**
3. Jede Datei ist mit Pillow lesbar; animierte GIFs werden abgewiesen. **[I]**
4. Ist das Referenzmanifest geladen (Standardmodus), darf kein Testbild pfadgleich oder **SHA-256-gleich** mit einem Referenz- oder Kontextbild des kuratierten Manifests sein; auch umbenannte Kopien werden so erkannt. **[I+T]** (`test_reference_copy_is_rejected_as_independent_test`)

**[E]** In Variante 1 wird das Referenzmanifest bewusst nicht geöffnet. Folglich entfällt dort auch der Abgleich der Testbilder gegen bekannte Referenzbilder (`known_dataset_paths` ist leer, [Tool/ki_bildanalyse.py:200-208](../Tool/ki_bildanalyse.py#L200-L208)). Für Variante 1 ist das hinsichtlich einer Übertragung von Referenzen unkritisch, für die Unabhängigkeit eines Evaluationsdatensatzes aber zu beachten (siehe [Offene_Punkte_und_Validierungsbedarf.md](Offene_Punkte_und_Validierungsbedarf.md)).

### 5.6 Bildübertragung

Für die Übertragung wird je Bild eine JPEG-Kopie **ausschließlich im Arbeitsspeicher** erzeugt: EXIF-Orientierung wird angewendet, das Bild in RGB umgewandelt und auf eine maximale Kantenlänge verkleinert (OpenAI und Mistral 2 048 px, Ollama standardmäßig 768 px), Qualität 90 ([Tool/ki_bildanalyse.py:346-362](../Tool/ki_bildanalyse.py#L346-L362)). Die Originaldateien werden nicht verändert. **[I+T]** Da Pillow beim Speichern ohne `exif`-Parameter keine EXIF-Daten schreibt, enthalten die übertragenen Kopien nach Codelage keine EXIF-Metadaten. **[I]** (nicht gesondert getestet)

Dateinamen, Prüfkörpernummern und Projektnamen werden **nicht** an das Modell übertragen; Testbilder werden nur als „Unbekanntes Testbild 1 … n“ angekündigt ([Tool/ki_bildanalyse.py:505-509](../Tool/ki_bildanalyse.py#L505-L509)). **[I+T]**

### 5.7 Speicherung von Dateipfaden und Bildidentitäten

Jede Ergebnisdatei enthält für jedes übertragene Bild Dateiname und SHA-256 der **Originaldatei** (`inputs`), die Listen der einbezogenen und der nicht ausgewählten Bilder (`image_selection.included/excluded`) sowie für jede verwendete Referenz Code, Dateiname, Prüfkörper-ID, absoluten Pfad und SHA-256 (`classification_basis.references`). Absolute Pfade der Testbilder werden nicht gespeichert; die Zuordnung erfolgt über Projekt, Prüfkörper-ID und Dateiname. **[I+T]**

## 6. Klassifikationssystem

### 6.1 Ist-Stand der Klassifikationsdateien

| Datei | Version / Stand | Inhalt | Rolle im Code |
|---|---|---|---|
| `versagensarten.json` | – | 29 Codes mit Kurzname | nicht vom Code geladen (Ursprungsliste) |
| `versagensarten_erweitert.json` | 1.0 / 20.10.2026 | 29 Klassen mit `feature_type`, `definition`, `visual_features`, `influence_on_failure`, `distinguishing_features`, `limitations`, `sources`, `review_required` | Pflichtbasis; `load_catalog` verlangt exakt 29 eindeutige Codes |
| `klassifikationsregeln.json` | **1.2 / 26.10.2026** | **acht Klassen** mit einheitlichen Pflichtfeldern | bestimmt die aktiven Klassen und den Promptinhalt |
| `Archiv/klassifikationsregeln_v1.1.json` | 1.1 / 25.10.2026 | sieben Klassen (ohne LR) | Archiv; SHA-256 in `meta.archiv_vorversion` hinterlegt und mit der Datei übereinstimmend |

**Ergebnis der Prüfung:** Der fachlich beschlossene Zielstand mit acht Klassen und Version 1.2 ist **im Code und in den JSON-Dateien tatsächlich umgesetzt**. Die aktiven Codes werden dynamisch aus der Reihenfolge der Einträge in `klassifikationsregeln.json` abgeleitet ([Tool/ki_bildanalyse.py:1228-1239](../Tool/ki_bildanalyse.py#L1228-L1239)); `run_analysis` verweigert jede abweichende Klassenliste. Der Test `test_version_1_2_has_eight_uniform_rules_and_archives_1_1` prüft Version, Klassenzahl und Archiv. **[I+T]**

Die Definitionen der übrigen 17 Klassen bleiben in `versagensarten_erweitert.json` erhalten, werden für neue Analysen aber nicht übertragen. Historische Ergebnisse mit 29 zulässigen Klassen bleiben lesbar.

**[E] Inkonsistenz in der Oberfläche:** Das Banner „Analysemodus“ im Register Bildanalyse lautet „Sieben dokumentierte Hauptversagensklassen sind aktiv; Variante 1 bleibt referenzfrei.“ ([frontend/src/App.tsx:102](../frontend/src/App.tsx#L102)), während die Fußzeile korrekt „8 aktive Hauptversagensklassen“ anzeigt. Der Banner-Text stammt aus dem Stand 1.1 und ist nicht angepasst. Ebenso nennt die Fehlermeldung in `validate_and_enrich` noch „eine der 29 Klassen“ ([Tool/ki_bildanalyse.py:978](../Tool/ki_bildanalyse.py#L978)).

### 6.2 Aufbau einer Klassenregel (Version 1.3)

Jede Regel enthält die Pflichtfelder `code`, `name`, `klassentyp`, `klassifikationsziel`, `positive_bildmerkmale`, `abgrenzung_zu_aehnlichen_klassen`, `nicht_vergeben_wenn`, `unklar_wenn`, `einschraenkungen_visuelle_erkennung`, `nicht_zulaessige_ursachenaussage`, `fachlicher_pruefstatus` (mit `status`, `begruendung`, `offene_punkte`) und `literaturquellen`. `load_catalog` prüft Vollständigkeit, nicht leere Textlisten und die Eindeutigkeit der Codes ([Tool/ki_bildanalyse.py:140-185](../Tool/ki_bildanalyse.py#L140-L185)). **[I+T]**

Die Metadaten (`meta`) enthalten Gültigkeitsbereich, Fotodokumentation, Grundprinzipien, die Bedeutung der Prüfstatuswerte und eine Quellenprüfung mit Zugriffsvermerk und Aussagegrenze je Quelle. Die Quellenprüfung ist Teil der Projektdatei; sie wurde für diesen Bericht nicht erneut nachvollzogen.

### 6.3 Die zwölf aktiven Merkmale

| Nr. | Code | Bezeichnung | Klassentyp | Fachlicher Prüfstatus |
|---|---|---|---|---|
| 1 | GSF | Globale Schrägfaser | sichtbares Holzmerkmal | bedingt geeignet |
| 2 | LSF | Lokale Schrägfaser | sichtbares Holzmerkmal | bedingt geeignet |
| 3 | KA | Kantenast | sichtbares Holzmerkmal | bedingt geeignet |
| 4 | SA | Seitenast | sichtbares Holzmerkmal | bedingt geeignet |
| 5 | sp HB | Spröder Holzbruch | sichtbares Bruchbild | fachlich offen |
| 6 | Spl HB | Splitterholzbruch | sichtbares Bruchbild | fachlich offen |
| 7 | RiSch | Ringschäle | sichtbares Holzmerkmal | bedingt geeignet |
| 8 | LR | Längsriss | sichtbares Holz- oder Bruchmerkmal | fachlich offen |
| 9 | Bu | Buchs | sichtbares Holzmerkmal | fachlich offen |
| 10 | AA | Astansammlung | sichtbares Holzmerkmal | fachlich offen |
| 11 | SSA | Schmalseitenast | sichtbares Holzmerkmal | fachlich offen |
| 12 | KB | Klemmbacken | sichtbare Prüf- oder Einspannspur | fachlich offen |

Die folgenden Kurzbeschreibungen geben den Inhalt von `klassifikationsregeln.json` sinngemäß wieder.

**GSF – Globale Schrägfaser.** Der erkennbare Faserverlauf weicht über einen größeren sichtbaren Abschnitt gleichgerichtet von der Prüfkörperlängsachse ab. Abgrenzung: LSF ist örtlich begrenzt; eine schräge Bruchkante allein ist kein Nachweis; Jahrring- oder Fladerzeichnung darf nicht automatisch als Faserrichtung gelesen werden. Unsicherheit: Die Draufsichten zeigen keine Faserneigung in Dickenrichtung; Mindestlänge für „global“ und zulässige Winkelstreuung sind laborintern nicht festgelegt.

**LSF – Lokale Schrägfaser.** Der Faserverlauf weicht nur in einem begrenzten Bereich ab, häufig um einen Ast; außerhalb der Störzone ist der Verlauf gerader. Abgrenzung: Ein Ast allein beweist keine LSF; LSF darf zusätzlich zu KA oder SA vergeben werden, wenn die Umlenkung sichtbar ist. Unsicherheit: Ausdehnung der lokalen Zone und eine mögliche Winkelregel sind nicht festgelegt; Winkelmessungen aus perspektivischen Fotos sind nicht belastbar.

**KA – Kantenast.** Der sichtbare Astanschnitt erreicht oder unterbricht eine Längskante in mindestens einer Draufsicht. Abgrenzung: SA liegt innerhalb der Breitseite; da keine Schmalseitenaufnahmen vorliegen, darf der Astverlauf auf der Schmalseite nicht als gesehen behauptet werden. Unsicherheit: Kante nicht vollständig im Bild, Kantenberührung nur durch Bildrand vermutet.

**SA – Seitenast.** Der Astanschnitt liegt innerhalb der Breitseite, zwischen Ast und beiden vollständig abgebildeten Längskanten liegt Holz. Unsicherheit: Ist der Randbereich nicht vollständig abgebildet, dürfen KA und SA nicht mit unbegründeter Sicherheit unterschieden werden.

**sp HB – Spröder Holzbruch.** Ausschließlich Bruchmorphologie: vergleichsweise kurze, überwiegend quer verlaufende, wenig zerfaserte Bruchzone. Ausdrücklich kein Nachweis mechanischer Sprödigkeit, die nur aus Kraft-Verformungs-Daten bestimmbar wäre. Die Klasse darf nicht allein aus einem fehlenden Spl-HB-Nachweis abgeleitet werden.

**Spl HB – Splitterholzbruch.** Stark zerfaserte, in Faserrichtung ausgedehnte Bruchzone mit langen Splittern, herausgezogenen Faserbündeln und Längsaufspaltungen. Aussagen wie „zähes“, „gesundes“ oder „hochfestes“ Holz sind unzulässig. Ein einzelner Längsriss ist kein Spl HB.

**RiSch – Ringschäle.** Trennung entlang oder annähernd entlang einer Jahrringgrenze, bei der sich ein zusammenhängender Holzbereich schicht- oder schalenartig löst. Ohne sichtbare Jahrringgrenze darf nur ein „ringschälenartiges“ Merkmal beschrieben werden; ob die Trennung vor oder während des Versuchs entstand, ist aus Nachbruchfotos nicht bestimmbar.

**LR – Längsriss.** Länglicher Riss mit erkennbaren Rissufern, überwiegend parallel zur lokalen Faserrichtung, innerhalb oder außerhalb der Bruchzone. Fachlich offen ist, ob LR im Labor als vorhandenes Holzmerkmal oder als versuchsbedingtes Bruchmerkmal geführt wird; Risstiefe und Entstehungszeitpunkt sind nicht bestimmbar.

### 6.4 Trennung von Merkmal, Bruchbild und Versagensursache

Das Klassifikationssystem unterscheidet drei Ebenen, die in den Regeln, im Prompt und in der Ergebnisstruktur getrennt geführt werden:

1. **Sichtbares Holzmerkmal** (GSF, LSF, KA, SA, RiSch, teilweise LR): Wuchsmerkmal oder Trennung, die am Prüfkörper erkennbar ist.
2. **Sichtbares Bruchbild** (sp HB, Spl HB, teilweise LR): Morphologie der Bruchzone.
3. **Versagensursache:** wird von der Software nicht bestimmt. Jede Regel enthält eine `nicht_zulaessige_ursachenaussage`; jedes ausgegebene Merkmal erhält im Ergebnis `proven_failure_cause: false` und einen festen Kausalitätshinweis ([Tool/ki_bildanalyse.py:1035-1039](../Tool/ki_bildanalyse.py#L1035-L1039)).

### 6.5 Mehrfachklassifikation und Unsicherheit

Das Ausgabeschema erlaubt einen primären Kandidaten und bis zu vier sekundäre Merkmale, jeweils mit Belegen und Konfidenz. Doppelte Codes werden abgewiesen. Der Prompt erklärt ausdrücklich, dass der primäre Kandidat **keine bestätigte Hauptversagensursache** ist. Die Regeln nennen Beispiele zulässiger Kombinationen (KA mit LSF, RiSch mit Spl HB). Bei unzureichender Evidenz ist `UNKLAR` auszugeben; dann müssen primärer Code `UNKLAR` und die Merkmalslisten leer sein. Die Unterscheidung „nicht sichtbar“ (nicht erkennbar) und „nicht beurteilbar“ (Bildqualität oder Perspektive erlauben keine Entscheidung) ist im Prompt verankert. Es besteht keine Pflicht, sp HB oder Spl HB zu vergeben; der Prompt sagt ausdrücklich „Erzwinge insbesondere weder sp HB noch Spl HB“. **[I+T]**

**[E]** Die Konfidenzwerte sind vom Modell selbst angegebene Zahlen zwischen 0 und 1. Sie sind nicht kalibriert und keine Wahrscheinlichkeiten im statistischen Sinn.

## 7. Prompt Engineering

### 7.1 Methodischer Ansatz

Die implementierte Prompt-Engineering-Methodik lässt sich als **regelbasiertes, schema-gebundenes und zweiphasiges Prompting** beschreiben. Sie besteht aus sechs Bausteinen:

1. **Rollen- und Grenzdefinition** im Systemprompt (nur sichtbare Merkmale, keine Bruchursache, striktes JSON).
2. **Zweiphasige Aufgabenstellung** im Benutzerprompt: erst neutrale Beobachtung, dann Klassenkandidaten.
3. **Wissensbasis als strukturierte Daten:** Die Klassenregeln werden als kompaktes JSON übertragen, nicht als Fließtext.
4. **Explizite Negativ- und Unsicherheitsregeln** je Klasse (`do_not_assign_when`, `unclear_when`, `visual_limitations`).
5. **Strukturierte Ausgabe** über ein JSON-Schema mit Aufzählungstypen für die Klassencodes, das der Provider erzwingen soll.
6. **Nachgelagerte Validierung** im Code, die Codes, Struktur, Konsistenz und unzulässige Kausalaussagen prüft.

Hinzu kommt eine **Informationsdiät**: Alles, was das Ergebnis vorwegnehmen könnte (Protokollursachen, Ground Truth, frühere Ergebnisse, Bewertungen, Literatur und interne Prüfvermerke), wird nicht übertragen (7.7).

### 7.2 Systemprompt

Der Systemprompt ist eine Konstante ([Tool/ki_bildanalyse.py:68-71](../Tool/ki_bildanalyse.py#L68-L71)):

> „Du sichtest ausschliesslich sichtbare Merkmale von Holz-Zugproben. Du gibst keine nachgewiesene Bruchursache aus und haeltst dich strikt an das JSON-Schema.“

Er wird bei OpenAI als `instructions`, bei Mistral und Ollama als Nachricht mit der Rolle `system` übertragen.

### 7.3 Benutzerprompt

Der Benutzerprompt entsteht in `_analysis_instructions` ([Tool/ki_bildanalyse.py:447-481](../Tool/ki_bildanalyse.py#L447-L481)) als JSON-Objekt mit den Feldern `task`, `rules` und `classes`. Die Regelliste enthält in der aktuellen Version (Promptversion `3.1-eight-class-visual-only`) unter anderem:

- zwei strikt getrennte Phasen (Beobachtung, danach Klassenvorschlag),
- Aufzählung der zu beschreibenden Merkmalsgruppen (Risse, Splitter/Auffaserung, mögliche Jahrringtrennungen, Äste und deren Lage, Faserrichtung, Bruchflächen),
- Sichtung des gesamten Prüfkörpers inner- und außerhalb der Bruchzone,
- Kennzeichnung unsicherer Beobachtungen; Unterscheidung „nicht sichtbar“/„nicht beurteilbar“,
- ausschließliche Verwendung der bereitgestellten Codes oder `UNKLAR`; die Anzahl wird dynamisch eingesetzt („8 Klassencodes“),
- Gleichberechtigung aller Klassen; kein Zwang zu sp HB oder Spl HB,
- Mehrfachklassifikation; der primäre Kandidat ist keine bestätigte Hauptversagensursache,
- Begründung ausschließlich mit sichtbaren Bildmerkmalen; Verbot von Ursachenbehauptungen,
- `UNKLAR` bei unzureichendem Bildmaterial,
- Verbot erfundener Winkel, Längen, Anzahlen oder Schwellen,
- Längenbegrenzung der Ausgabe und Konsistenzregel für `UNKLAR`.

Im Standardmodus (mit Referenzen) kommen drei Regeln hinzu: Klassen ohne Referenzbild bleiben zulässig, Referenzbilder sind nur Beispiele, und in der Ausgabe dürfen nur die Testbilder beschrieben werden. In Variante 1 ersetzt die Regel „Beurteile ausschließlich die ausgewählten unbekannten Prüfkörperbilder und die bereitgestellten Klassenbeschreibungen“ diese drei Regeln.

### 7.4 Einbindung der Klassifikationsregeln

`_prompt_catalog` ([Tool/ki_bildanalyse.py:407-444](../Tool/ki_bildanalyse.py#L407-L444)) überführt jede aktive Regel in ein kompaktes englischsprachig benanntes Objekt mit deutschen Inhalten:

| Promptfeld | Quelle in `klassifikationsregeln.json` |
|---|---|
| `code`, `name` | `code`, `name` |
| `definition` | `klassifikationsziel` |
| `positive_image_features` | `positive_bildmerkmale` |
| `distinguishing_features` | `abgrenzung_zu_aehnlichen_klassen` |
| `uncertainty_rules.do_not_assign_when` | `nicht_vergeben_wenn` |
| `uncertainty_rules.unclear_when` | `unklar_wenn` |
| `uncertainty_rules.visual_limitations` | `einschraenkungen_visuelle_erkennung` |
| `confirmed_reference_count` | nur im Standardmodus: Zahl der bestätigten Hauptreferenzen der Klasse |

**Nicht** übertragen werden `klassentyp`, `nicht_zulaessige_ursachenaussage`, `fachlicher_pruefstatus` (inklusive offener Punkte) und `literaturquellen` sowie die gesamten Metadaten. Der Test `test_variant_one_prompt_contains_only_eight_compact_descriptions_and_test_images` sichert diese Auswahl ab. **[I+T]**

Für Klassen ohne Regel (nur relevant für historische 29-Klassen-Läufe) würde stattdessen die erweiterte Beschreibung aus `versagensarten_erweitert.json` verwendet ([Tool/ki_bildanalyse.py:431-440](../Tool/ki_bildanalyse.py#L431-L440)).

Für die Rückverfolgbarkeit wird der SHA-256 der tatsächlich verwendeten kompakten Klassenbeschreibungen als `active_class_descriptions_sha256` gespeichert. Für den Stand 1.2 lautet er `13d916d3b7e340ceea177c070268036c445c7927d3fa71cd9a4a99e8321d16ca` (lokal berechnet am 26.10.2026). Der Benutzerprompt umfasst in Variante 1 rund 16 400 Zeichen (kompaktes JSON).

### 7.5 Einbindung der Bilder und Nachrichtenreihenfolge

Die Reihenfolge ist für alle Provider gleich ([Tool/ki_bildanalyse.py:484-517](../Tool/ki_bildanalyse.py#L484-L517), [786-828](../Tool/ki_bildanalyse.py#L786-L828)):

1. Benutzerprompt (JSON),
2. nur im Standardmodus: je Referenz der Text „Bestaetigtes Hauptreferenzbild fuer Klasse <Code>.“ und das Bild,
3. Trenntext „Ab hier folgen ausschliesslich ausgewaehlte Bilder des unbekannten Testpruefkoerpers.“,
4. je Testbild der Text „Unbekanntes Testbild <n>.“ und das Bild,
5. Abschlussanweisung „Erzeuge jetzt ausschliesslich die knappe JSON-Klassifikation aller bereitgestellten Testbilder. Beschreibe keine Referenzbilder.“

### 7.6 Ausgabeformat, Strukturierung und Validierung

Das Schema (`response_schema`, [Tool/ki_bildanalyse.py:520-549](../Tool/ki_bildanalyse.py#L520-L549)) verlangt die Felder `status` (`KLASSIFIZIERT` | `UNKLAR`), `primary_code` (aktive Codes oder `UNKLAR`), `primary_evidence`, `primary_confidence` (0–1), `secondary_features` (höchstens 4 Objekte mit `code`, `evidence` [≤ 3], `confidence`), `visible_observations` (≤ 6), `uncertainty_reasons` (≤ 4) und `missing_views` (≤ 4); zusätzliche Felder sind nicht erlaubt.

`validate_and_enrich` ([Tool/ki_bildanalyse.py:959-1014](../Tool/ki_bildanalyse.py#L959-L1014)) prüft unabhängig vom Provider erneut:

- gültiger Status und gültige Codes (auch sekundär), keine Mehrfachnennung,
- Konsistenz von `UNKLAR` (kein primärer oder sekundärer Code),
- bei `KLASSIFIZIERT` mindestens ein Beleg für den primären Kandidaten und jeden sekundären Kandidaten,
- Konfidenz numerisch zwischen 0 und 1,
- **Kausalitätsfilter:** Belege, Beobachtungen, Unsicherheiten und fehlende Ansichten dürfen keine Formulierungen wie „verursacht“, „ursächlich“, „Bruchursache“, „führte zum“ oder „löste den Bruch aus“ enthalten ([Tool/ki_bildanalyse.py:72-75](../Tool/ki_bildanalyse.py#L72-L75)). Ein Treffer führt zur Zurückweisung der gesamten Antwort. **[I+T]**

Anschließend wird jedes Merkmal um Name, Merkmalstyp, Referenzbildanzahl, `proven_failure_cause: false` und einen Kausalitätshinweis ergänzt.

**[E]** Der Kausalitätsfilter erfasst nur die genannten deutschen Wortformen; englische oder umschreibende Kausalformulierungen werden nicht erkannt. Die Längenbegrenzungen des Schemas werden im Code nicht erneut geprüft, sondern vom Provider erwartet.

### 7.7 Mechanismen zur Vermeidung unzulässiger Informationen

| Information | Behandlung | Nachweis |
|---|---|---|
| Versuchsprotokolle, dokumentierte Bruchursachen | werden nicht geladen | im Code kein Lesezugriff; Ergebnisfeld `protocol_failure_cause_transmitted: false` |
| Ground Truth | wird von `ki_bildanalyse` nicht gelesen | Ergebnisfeld `ground_truth_transmitted: false` |
| frühere KI-Ergebnisse, Expertenbewertungen | werden nicht gelesen | Felder `previous_results_transmitted`, `expert_assessments_transmitted: false`; Test `test_repeated_runs_are_independent_and_log_reproducible_inputs` |
| Prüfkörpernummer, Dateinamen, Pfade | nicht im Prompt | Test zum Promptinhalt |
| Literatur, Prüfstatus, offene Punkte der Regeln | nicht im Prompt | `_prompt_catalog` |
| Referenzbilder in Variante 1 | Manifest wird nicht geöffnet | Test `test_variant_one_catalog_does_not_open_reference_manifest` |
| Kontextbilder (u. a. Nummernfotos der Referenzprüfkörper) | nie als Beispiel | Auswahl nur `reference_image` mit `default_for_inference=true` |

Die Felder mit dem Wert `false` sind im Ergebnisdokument fest gesetzt; sie dokumentieren das Codeverhalten und sind kein unabhängiger Laufzeitnachweis.

### 7.8 Unterschiede zwischen Qwen, OpenAI und Mistral

Der **fachliche Promptinhalt ist für alle drei Provider identisch** (Tests `test_openai_and_ollama_share_the_same_factual_prompt`, `test_mistral_uses_the_same_factual_prompt_and_content_order`). **[I+T]** Unterschiede bestehen nur im Transport und in Parametern:

| Aspekt | Qwen3-VL über Ollama | OpenAI | Mistral |
|---|---|---|---|
| Nachrichtenform | je Text/Bild eine eigene `user`-Nachricht, Bilder als Base64 im Feld `images` | eine `user`-Nachricht mit Liste aus `input_text`/`input_image` (Data-URL, `detail: high`) | eine `user`-Nachricht mit `text`/`image_url` |
| Bildkante | 768 px (Standard) | 2 048 px | 2 048 px |
| Strukturierte Ausgabe | `format` = JSON-Schema | `text.format` `json_schema`, `strict: true` | `response_format` `json_schema`, `strict: true` |
| Temperatur | 0 | nicht gesetzt (Provider-Standard) | 0 |
| Maximale Ausgabe | `num_predict` 4 096 | nicht gesetzt | `max_tokens` 4 096 |
| Kontext | `num_ctx` 24 576 | – | – |
| Weitere Parameter | `think: false`, `stream: false` | `store: false` | `stream: false` |
| Bilder je Anfrage | unbegrenzt im Code | unbegrenzt im Code | höchstens 8 (Referenzen + Testbilder); Aufteilung in Teilanfragen |
| Antwort-Parsing | tolerant: entfernt Markdown-Codeblöcke, sucht äußerste geschweifte Klammern, erlaubt Steuerzeichen | `json.loads(response.output_text)` | `json.loads` des Nachrichtentexts |

**[E]** Bei OpenAI wird keine Temperatur gesetzt; die Läufe sind daher nicht auf deterministische Dekodierung eingestellt. Bei Mistral werden im Standardmodus sieben Referenzen übertragen, sodass je Anfrage nur ein Testbild Platz hat; die Teilergebnisse werden zusammengeführt (8.4). Das Modell sieht dann den Prüfkörper nie in mehreren Ansichten gleichzeitig. Dies ist ein methodischer Unterschied zum Vergleich mit Qwen und OpenAI.

### 7.9 Promptversionierung und Prompt-Hash

- `PROMPT_VERSION = "3.1-eight-class-visual-only"` ([Tool/ki_bildanalyse.py:67](../Tool/ki_bildanalyse.py#L67)) wird in jedem Ergebnis gespeichert. Historische Ergebnisse tragen die Version `2.0` oder keine Angabe.
- Der Prompt-Hash ist der SHA-256 eines kanonisch sortierten JSON-Manifests aus Promptversion, Systemprompt, Benutzerprompt, Antwortschema, Dateiname und SHA-256 jedes Testbilds sowie den Referenzdatensätzen ([Tool/ki_bildanalyse.py:1409-1423](../Tool/ki_bildanalyse.py#L1409-L1423)). **[I+T]**
- Zusätzlich werden die Regelversion (`classification_rules_version`, `active_class_set_version`) und der Hash der Klassenbeschreibungen gespeichert.

**[E]** Der Prompt-Hash umfasst weder Provider, Modell und Parameter noch die tatsächlich übertragenen, verkleinerten Bildbytes. Bei identischen Eingaben haben Qwen-, OpenAI- und Mistral-Läufe denselben Prompt-Hash; das ist für den Modellvergleich erwünscht, muss bei der Interpretation aber bekannt sein. Die Promptversion ist eine manuell gepflegte Konstante; eine automatische Erhöhung bei Codeänderungen gibt es nicht.

### 7.10 Implementierter Prompt, wissenschaftliche Zieldefinition und Verbesserungspotenzial

| Ebene | Inhalt |
|---|---|
| **Tatsächlich implementiert** | Systemprompt, zweiphasiger Benutzerprompt mit zwölf kompakten Regeln, Bildreihenfolge, JSON-Schema, Validierung, Kausalitätsfilter, Promptversion und Hash (7.2–7.9). Die Beispieldatei `Datenbank/klassifikationsprompt_v1.3_beispiel.json` stimmt mit dem aus dem Code erzeugten Prompt überein (lokal geprüft). |
| **Wissenschaftliche Zieldefinition** | Das Modell beschreibt sichtbare Merkmale, schlägt nachvollziehbar begründete Klassen vor, kennzeichnet Unsicherheit und überlässt die Ursachenbestimmung der Fachperson. Die Bedingungen der Varianten sollen sich nur in der jeweils definierten Zusatzinformation unterscheiden (Kapitel 9). |
| **Verbesserungspotenzial** | (a) Die Zweiphasigkeit ist im Standardpfad nur durch eine Anweisung, nicht technisch erzwungen; die Antwort enthält Beobachtungen und Kandidaten in einem Aufruf. Variante D des Evaluationswerkzeugs zeigt eine technisch getrennte Alternative. (b) Kausalitätsfilter auf weitere Formulierungen und Englisch erweitern. (c) Temperatur bei OpenAI festlegen oder ausdrücklich begründen. (d) Promptversion automatisch an einen Hash des Promptgenerators koppeln. (e) Bezug von Beobachtungen zu einzelnen Testbildern (Bildindex) in das Schema aufnehmen. (f) Fehlende Ansichten (Stirnholz, Schmalseite) als feste Liste statt Freitext erfassen. (g) Für Mistral eine Variante ohne Teilanfragen prüfen. |

## 8. KI-Modelle und Modellanbindung

### 8.1 Übersicht

Ollama ist die lokale Laufzeitumgebung, Qwen3-VL 4B das darin betriebene Modell. Die aktive Modellauswahl umfasst genau drei Kombinationen:

| Anzeige in der App | Provider | Modellkennung | Dokumentation im Projekt |
|---|---|---|---|
| Qwen3-VL 4B · lokal | `ollama` | `qwen3-vl:4b` | keine Verifikationsdatei |
| OpenAI GPT-6.1 · Cloud | `openai` | `gpt-6.1-sol` | `Tool/model_verification.json` (Stand 21.10.2026, `status: documented_candidate`, `account_access: not_verified_without_api_call`) |
| Mistral Ministral 3B · Cloud | `mistral` | `ministral-3b-2512` | `Tool/mistral_model_verification.json` (Stand 22.10.2026, `status: documented_candidate`, Kontext laut Datei 256 000 Token) |

LLaVA (`llava:latest`) ist aus der Auswahl entfernt; ein Startversuch wird abgewiesen. Ein historisches LLaVA-Ergebnis liegt unter `Output/KI_Klassifikation/Testlauf_5_Pruefkoerper/Ollama_Ergebnisse/` und bleibt lesbar (Test `test_historical_llava_result_remains_readable_and_exportable`). **[I+T]**

Vor jedem OpenAI- oder Mistral-Lauf prüft der Code, ob das Modell in der jeweiligen Verifikationsdatei mit Bild- und API-Fähigkeit eingetragen ist; andernfalls bricht er ab ([Tool/ki_bildanalyse.py:552-573](../Tool/ki_bildanalyse.py#L552-L573)). **[I+T]**

### 8.2 Ollama / Qwen3-VL (lokal)

- Endpunkt fest `http://localhost:11434/api/chat`; andere Endpunkte werden abgewiesen. **[I+T]**
- Payload: `model`, `messages`, `format` (JSON-Schema), `stream: false`, `think: false`, `options` mit `temperature: 0`, `num_predict`, `num_ctx`.
- Zeitlimit 900 s; Zeitüberschreitung mit verständlicher Meldung. **[I+T]**
- Fehlerfälle: HTTP-Fehler (gekürzter Body), Dienst nicht erreichbar, unlesbares JSON. Bei unlesbarer Antwort werden Zeichenzahl, Zahl der „Thinking“-Zeichen, `prompt_eval_count`, `eval_count` und `done_reason` gemeldet, um z. B. ein erschöpftes Generationsbudget zu erkennen. **[I+T]**
- Metadaten: Laufzeiten und Tokenzähler von Ollama, gemeldete Modellkennung, verwendete Optionen.
- Die App startet Ollama nicht und lädt keine Modelle herunter; der Dienst muss bereits laufen.

### 8.3 OpenAI

- Schlüssel aus `OPENAI_API_KEY`, ersatzweise aus dem macOS-Schlüsselbunddienst `KI-Bemessungstool OpenAI` (`security find-generic-password`, Zeitlimit 3 s). Der Schlüssel wird nicht protokolliert (`repr=False`). **[I+T]**
- Aufruf `client.responses.create(model, instructions, input, text.format=json_schema strict, store=False)`.
- Metadaten: Response-ID, gemeldete Modellkennung, Tokenverbrauch; ein Rechnungsbetrag wird nicht gemeldet (`reported_amount: null`).
- Ein explizites Zeitlimit wird nicht gesetzt; es gilt der Standard des SDK. **[E]**
- Praktischer Nachweis: Im Projekt `Test_001` liegen sechs historische Ergebnisse mit `mode: live` (21.–24.10.2026). **[I+P]** Für die aktuelle Promptversion 3.1 liegt kein Lauf vor.

### 8.4 Mistral

- Schlüssel **ausschließlich** aus dem macOS-Schlüsselbunddienst `Usecase_2_Mistral`; keine Umgebungsvariable, auf anderen Betriebssystemen nicht konfigurierbar. **[I+T]**
- Direkter HTTPS-Aufruf ohne SDK an den fest vorgegebenen Endpunkt; Payload mit `temperature: 0`, `max_tokens: 4096`, `stream: false`, `response_format.json_schema` (Feldname `schema`, `strict: true`).
- Höchstens acht Bilder je Anfrage. Referenzen werden auf sieben begrenzt; die Testbilder werden in Gruppen der verbleibenden Größe übertragen. Jede Teilantwort wird einzeln validiert; anschließend übernimmt `_merge_batch_results` den Kandidaten mit der höchsten Konfidenz als primär, fasst sekundäre Merkmale (höchste Konfidenz je Code, höchstens vier) und Textlisten (dedupliziert, gekürzt) zusammen ([Tool/ki_bildanalyse.py:1043-1088](../Tool/ki_bildanalyse.py#L1043-L1088)). **[I+T]**
- Fehlermeldungen lesen den HTTP-Body und schwärzen Schlüssel, Bearer-Token und Bilddaten. **[I+T]**
- Praktischer Nachweis: sechs historische Ergebnisse mit `mode: live` (23.–24.10.2026). **[I+P]** In `Tool/README.md` steht, dass nach der Korrektur des HTTP-422-Fehlers (Feld `schema` statt `schema_definition`) kein Live-Aufruf mehr erfolgt sei. Die vorhandenen Ergebnisdateien mit Zeitstempeln nach dem Mock-Test der Korrektur (`Mistral_422_Fix_Mock`, 23.10.2026 06:26 UTC) widersprechen dieser Aussage. Ob die README veraltet ist, muss die Anwenderin bestätigen.

### 8.5 Mock-Modus

Ohne `--live` werden OpenAI und Mistral in der Kommandozeile nicht aufgerufen; stattdessen wird `Tool/mock_response.json` (Status `UNKLAR`) validiert und gespeichert, ohne den Schlüsselbund zu lesen. **[I+T]** In der App werden Cloud-Modelle dagegen immer live aufgerufen, sobald die Kostenbestätigung erteilt ist.

**[E]** `Tool/README.md` beschreibt den Prototyp als „standardmäßig ausschließlich mit einer lokalen Mock-Antwort“. Der Standard-Provider der Kommandozeile ist jedoch `ollama`; ein Aufruf ohne `--provider` sendet die Bilder an die lokale Ollama-Instanz. Kosten oder Cloud-Übertragungen entstehen dadurch nicht.

### 8.6 Technische Unterschiede zwischen lokalen und cloudbasierten Modellen

| Kriterium | Lokal (Qwen3-VL 4B) | Cloud (OpenAI, Mistral) |
|---|---|---|
| Datenfluss | Bilder verlassen den Mac nicht | Bilder werden an externe Anbieter übertragen |
| Kosten | keine API-Kosten | kostenpflichtig; Freigabe erforderlich |
| Laufzeit (historische Läufe, nur zur Orientierung) | ca. 109–518 s je Lauf | ca. 5–31 s je Lauf |
| Bildauflösung | 768 px | 2 048 px |
| Modellgröße | 4 Milliarden Parameter laut Modellbezeichnung | vom Anbieter nicht offengelegt bzw. im Projekt nicht dokumentiert |
| Reproduzierbarkeit | Modellversion lokal fixierbar | Modellstand liegt beim Anbieter; Änderungen sind nicht kontrollierbar |
| Ausgabeformat | Schema per `format`, toleranter Parser | strikte Schemas |

Die Laufzeiten stammen aus den historischen Ergebnisdateien (unterschiedliche Prompt- und Referenzstände) und sind **keine Leistungskennzahlen**.

## 9. Evaluationskonzept

### 9.1 Überblick über die implementierten Varianten

Im Projekt existieren zwei getrennte Variantensysteme:

1. **Varianten 1, 2a, 2b, 3** in der Desktop-App und in `ki_bildanalyse.py`. Davon ist **nur Variante 1 technisch definiert**. Für 2a, 2b und 3 lehnt `run_analysis` den Start mit der Meldung „Nur Evaluationsvariante 1 ist technisch definiert; die Zuordnung von 2a, 2b und 3 ist offen.“ ab ([Tool/ki_bildanalyse.py:1224-1227](../Tool/ki_bildanalyse.py#L1224-L1227)); die App zeigt sie als „Zuordnung offen“ an. **[G]**
2. **Varianten A, B, C, D** im Kommandozeilenwerkzeug `Tool/evaluation.py`. Sie sind implementiert und mit Fake-Transport getestet, aber **noch nie ausgeführt** worden (es existieren nur drei Laufpläne ohne Ergebnisse vom 22.10.2026, erstellt mit Regelversion 1.0 und 26 Fällen). **[I+T]**

Eine Zuordnung der Bezeichnungen 1/2a/2b/3 zu A–D oder zu bestimmten Referenzbildanzahlen ist im Projekt nicht festgelegt; `Evaluation/README.md` vermerkt dies ausdrücklich als offen. Dieser Bericht nimmt keine Zuordnung vor.

Zusätzlich gibt es den **Standardmodus** (keine Variante), der in der App für Einzel- und Batchanalysen verwendet wird.

### 9.2 Variante 1 (referenzfrei)

| Aspekt | Implementierung |
|---|---|
| Eingabedaten | ausgewählte Prüfkörperbilder (ohne Nummernfotos) |
| Promptinformation | Analyseanweisungen und kompakte Beschreibungen der zwölf aktiven Merkmale aus `klassifikationsregeln.json` v1.3 |
| Referenzbilder | keine; das Referenzmanifest wird nicht geöffnet, `max_references_per_class` wird auf 0 gesetzt |
| Ground Truth, frühere Ergebnisse, Expertenkorrekturen | nicht enthalten |
| Auswahlverfahren | Bildauswahl des Prüfkörpers (Vorauswahl bzw. manuelle Auswahl) |
| Schutzmechanismen | (1) Kontrollübersicht vor jedem Lauf; (2) einmal verwendbares Plan-Token, das exakt zu Prüfkörper, sortierter Bildliste, Provider, Modell und Klassenliste passen muss; (3) Klassenliste muss exakt den dokumentierten Regeln entsprechen; (4) in der App nur mit Qwen3-VL über Ollama |
| Ausgabe | Ergebnisdokument Schema 2.0 mit `evaluation_variant: "1"`, `reference_database_accessed: false`, `confirmed_main_reference_count_used: 0`, leerer Referenzliste |
| Protokollierung | Regelversion, Hash der Klassenbeschreibungen, Promptversion und -hash, Bild-Hashes, Laufzeit, Ollama-Optionen und Tokenzähler |
| Reproduzierbarkeit | gleiche Eingaben erzeugen denselben Prompt-Hash; die Modellausgabe selbst ist trotz Temperatur 0 nicht garantiert identisch (Kapitel 14) |
| Nachweisstand | **[I+T]** (Tests für Katalog, Kontrollübersicht, Token-Pflicht, Mock-Lauf und Promptinhalt). **Ein echter Lauf mit Variante 1 liegt nicht vor.** |

In der Kommandozeile ist Variante 1 mit `--evaluation-variant 1` für alle Provider verfügbar (Cloud-Provider weiterhin nur mit `--live --confirm-paid-request`); die Kontrollübersicht und die Qwen-Beschränkung gelten nur in der App.

Die Anforderung, dass die bloße Existenz der Referenzbilddatenbank nicht zur Übertragung von Referenzbildern in Variante 1 führen darf, ist damit **im Code umgesetzt**: Referenzen werden nur geladen, wenn `include_references=True`, und in Variante 1 wird `load_catalog(include_references=False, …)` aufgerufen. Der App-Referenzspeicher (`Referenzbilder_neu`) ist ohnehin nicht an die Modellaufrufe angebunden.

### 9.3 Standardmodus der App

| Aspekt | Implementierung |
|---|---|
| Promptinformation | wie Variante 1, zusätzlich Referenzregeln und `confirmed_reference_count` je Klasse |
| Referenzbilder | eine bestätigte Hauptreferenz je aktiver Klasse aus dem kuratierten Manifest, ohne Referenzen desselben Prüfkörpers und ohne hashgleiche Bilder. Beim aktuellen Manifest sind dies sieben Bilder (GSF, LSF, KA, SA, sp HB, Spl HB, RiSch); LR hat keine Referenz. Die Auswahl erfolgt deterministisch in Manifestreihenfolge (erste passende Referenz je Klasse). |
| Einsatz | alle Einzelanalysen im Register Bildanalyse und alle Batches |
| Nachweisstand | **[I+T]**; historische Läufe **[I+P]**, jedoch alle mit älteren Ständen (29 Klassen, 0–11 Referenzen) |

### 9.4 Varianten A–D (`Tool/evaluation.py`)

| Variante | Eingabe | Prompt | Referenzbilder | Modellaufrufe |
|---|---|---|---|---|
| A | Testbild(er) | Minimalanweisung + erlaubte Codes | keine | 1 |
| B | Testbild(er) | Minimalanweisung + Codes + erweiterte Beschreibungen aus `versagensarten_erweitert.json` (bewusst ohne die Entscheidungsregeln) | keine | 1 |
| C | Testbild(er) | Minimalanweisung + Codes | bestätigte Hauptreferenzen anderer Prüfkörper, max. n je Klasse (Standard 1) | 1 |
| D | Testbild(er) | Stufe 1: neutrale Merkmalsextraktion ohne Klassenwissen; Stufe 2: regelgestützte Klassifikation nur aus den Beobachtungen der Stufe 1, **ohne erneuten Bildzugriff** | keine | 2 |

Quelle: [Tool/evaluation.py:262-380](../Tool/evaluation.py#L262-L380). **[I+T]**

- **Datenbasis:** `Evaluation/ground_truth_manifest.json` enthält 26 Fälle, abgeleitet ausschließlich aus bestätigten Hauptreferenzbildern (bestätigt durch „CZ“ am 21.10.2026). Für die ursprünglichen acht Klassen werden 18 Fälle verwendet (GSF 3, LSF 3, KA 2, SA 3, sp HB 3, Spl HB 3, RiSch 1, LR 0). Testbild jedes Falls ist das jeweilige Hauptreferenzbild selbst.
- **Leave-one-Prüfkörper-out:** In Variante C werden alle Referenzen desselben Prüfkörpers und hashgleiche Bilder ausgeschlossen; eine nachträgliche Prüfung löst bei Leakage einen Fehler aus. **[I+T]**
- **Integritätsprüfung:** Ground Truth ist an den SHA-256 des Referenzmanifests gebunden; jedes Bild wird erneut gehasht. Ein Lauf bricht ab, wenn sich etwas geändert hat. Zum Analysezeitpunkt stimmt der Manifest-Hash (`a6c0847e…eb0`) überein.
- **Laufplan:** Ohne `--execute` wird nur ein Plan mit SHA-256 von Ground Truth, Evaluationscode, Klassendefinitionen, Regeln und Manifest geschrieben. Cloud-Läufe erfordern `--execute --live --confirm-paid-request`. **[I+T]**
- **Umfang:** Je Fall fünf Modellaufrufe (A, B, C je 1, D 2), bei 18 Fällen also 90 Aufrufe.
- **Ausgabe und Metriken:** `evaluation_results.json/.csv`, `confusion_matrices.csv`; je Variante Gesamtgenauigkeit (Anteil korrekter Primärklassen), klassenweise Precision, Recall und F1, Macro-F1 über die im Datensatz vertretenen Klassen, Verwechslungsmatrix; Fehler getrennt als `wrong_classification`, `unclassifiable`, `invalid_json`, `api_error`. `UNKLAR` und Fehler zählen als nicht korrekt. **[I+T]**

**[E]** Die Variante B nutzt die erweiterten Beschreibungen (Stand 1.0), nicht die Regeln 1.2; Variante D Stufe 2 nutzt die Regeln 1.2. Die Varianten unterscheiden sich daher nicht nur in einem Faktor. In Variante C bei Mistral wird die tatsächlich gekürzte Referenzliste im Datensatz nicht als solche protokolliert (`reference_count` zählt die ungekürzte Liste). Die Variante D erzwingt in Stufe 1 den Status `UNKLAR`, was das gemeinsame Schema missbraucht und bei Modellen, die dennoch klassifizieren, zum Fehler führt.

### 9.5 Wissenschaftliches Ziel des Modellvergleichs

Das Ziel ist zu untersuchen, ob und unter welchen Informationsbedingungen lokale und cloudbasierte Vision-Language-Modelle sichtbare Holzmerkmale und Bruchbilder an Zugprüfkörpern so beschreiben und vorschlagen können, dass sie die fachliche Dokumentation unterstützen. Die Varianten sollen den Einfluss zusätzlicher Information (Klassenbeschreibung, Regeln, Referenzbilder) isolieren.

**Vergleichbarkeit der Ausgangsbedingungen** ist im Code durch identischen Promptinhalt, identische Bildauswahl, identisches Schema und vollständige Hash-Protokollierung angelegt. Einschränkend wirken die unterschiedlichen Bildauflösungen (768 px vs. 2 048 px), die fehlende Temperaturfestlegung bei OpenAI und die Teilanfragen bei Mistral (7.8).

### 9.6 Mögliche Bewertungsmetriken und Mehrfachklassifikation

Implementiert sind nur Metriken für die **Primärklasse** (9.4). Für eine wissenschaftliche Auswertung der Mehrfachklassifikation sind zusätzlich denkbar, aber **nicht implementiert**: Multi-Label-Metriken (z. B. Hamming-Verlust, Jaccard-Index, Precision/Recall je Klasse über alle vergebenen Labels), getrennte Bewertung von „Merkmal korrekt erkannt“ und „Hauptversagen korrekt vorgeschlagen“, Rate begründeter `UNKLAR`-Antworten, Rate unzulässiger Kausalaussagen, Übereinstimmung wiederholter Läufe und Interrater-Übereinstimmung zwischen Fachpersonen. **[G]**

### 9.7 Rolle der fachlichen Ground Truth

Die Ground Truth ist die von einer Fachperson bestätigte Zuordnung eines Bildes zu einer Klasse. Im Evaluationswerkzeug ist sie auf bestätigte Hauptreferenzen beschränkt; KI-Vorschläge, Kontextbilder und Protokollklassen sind ausdrücklich ausgeschlossen. In der App werden die Expertenbewertungen (Kapitel 11) gespeichert und exportiert, sie fließen aber **nicht** automatisch in eine Ground-Truth-Datei oder in Metriken ein.

### 9.8 Stand der durchgeführten KI-Versuche

| Versuchsart | Vorhanden | Stand |
|---|---|---|
| App-Läufe Projekt `Test_001` | 19 Ergebnisdateien (Qwen3-VL 7, OpenAI 6, Mistral 6) | Schema 1.0/2.0, 29 zulässige Klassen, 0–11 Referenzen, Promptversion 2.0 oder ohne Angabe |
| Fachliche Bewertungen | 3 Bewertungsdateien (Prüfkörper Z-30-19), 1 Korrektur im Altformat (Z-30-06) | Funktionsnachweis, keine Stichprobe |
| Batches | 5 (je 1 Auftrag, Z-30-19) | 3 abgeschlossen, 2 unterbrochen/abgebrochen |
| CLI-Testlauf 5 Prüfkörper | Mock-, LLaVA-, Qwen-, OpenAI- und Mistral-Ergebnisse | Regelversion 1.0, 29 Klassen |
| Evaluation A–D | 3 Laufpläne | nicht ausgeführt |
| Variante 1 | – | **kein Lauf** |

Aus diesen Dateien werden in diesem Bericht bewusst **keine Trefferquoten** abgeleitet: Sie wurden mit veralteten Klassen- und Promptständen erzeugt, betreffen wenige Prüfkörper und sind nicht nach einem Evaluationsdesign geplant worden.

### 9.9 Methodische Grenzen

- Die Ground Truth stammt aus derselben Bildquelle wie die Referenzbilder (interne Evaluation, keine unabhängige externe Validierung).
- Kleine Fallzahlen je Klasse (1–3), eine Klasse (LR) ohne bestätigten Fall.
- Je Fall nur ein Testbild, während die App mehrere Ansichten überträgt.
- Fachlicher Prüfstatus der Regeln teils „fachlich offen“.
- Die Zuordnung 1/2a/2b/3 und die zugehörigen Referenzbildanzahlen sind nicht festgelegt.

## 10. Referenzbilddatenbank

### 10.1 Zwei getrennte Referenzbestände

| Bestand | Speicherort | Pflege | Verwendung |
|---|---|---|---|
| **Kuratierte Referenzen** | `Datenbank/Referenzbilder/` mit `referenzbilder_manifest.json` (Schema 2.0, 21.10.2026) | außerhalb der App erstellt (Bestätigungsquelle `Output/Referenzbilddatenbank_Gesamt/Bestaetigung_Gesamt.json`) | **Modellinput im Standardmodus**, Ground Truth, Evaluation A–D; zusätzlich im App-Bundle enthalten |
| **App-Referenzspeicher** | `Datenbank/Referenzbilder_neu/` | im Register „Referenzdatenbank“ | Verwaltung und Sichtung; **kein Modellinput**, keine Ground Truth |

Diese Trennung ist für die Interpretation wichtig: Bilder, die in der App als Referenz importiert und bestätigt werden, werden nach aktuellem Code **nicht** an Modelle übertragen.

### 10.2 Kuratierte Referenzen

Das Manifest enthält 26 Referenzsätze mit je einem Hauptreferenzbild (`reference_image`) und zwei bis vier Kontextbildern (`context_images`, darunter Nummernfotos und weitere Ansichten). Je Satz sind Klasse, Prüfkörper (Serien Z-A bis Z-G), Datensatzrolle `reference`, `excluded_from_independent_test: true`, Bestätigungsperson, Datum, Sichtbarkeitsstatus `SICHTBAR_BESTAETIGT`, Original- und Kopiepfad dokumentiert. Verteilung: SSA 3, SA 3, AA 3, LSF 3, GSF 3, sp HB 3, Spl HB 3, KA 2, Bu 1, HB 1, RiSch 1. Für die ursprünglichen acht Klassen stehen damit 18 Hauptreferenzen zur Verfügung; LR hat keine.

Als Modellinput werden nur Hauptreferenzbilder verwendet, die alle Freigabekriterien erfüllen (`dataset_role=reference`, `excluded_from_independent_test=true`, `image_role=reference_image`, `confirmed=true`, `default_for_inference=true`, `visibility=SICHTBAR_BESTAETIGT`); Kontextbilder nie. Ein fehlendes Hauptreferenzbild führt zum Abbruch ([Tool/ki_bildanalyse.py:222-244](../Tool/ki_bildanalyse.py#L222-L244)). **[I+T]**

**Ausschluss desselben Prüfkörpers:** `select_references` überspringt Referenzen, deren Prüfkörperordner (zweite Ebene über der Bilddatei) denselben Namen wie der analysierte Prüfkörperordner hat, sowie Referenzen, deren SHA-256 einem Testbild entspricht ([Tool/ki_bildanalyse.py:371-392](../Tool/ki_bildanalyse.py#L371-L392)). **[I+T]** Da Referenzen (Serien Z-A bis Z-G) und App-Prüfkörper (Serie Z-30) unterschiedlich benannt sind, greift der Namensausschluss in der Praxis nur bei gleichnamigen Ordnern; der Hashabgleich ist der wirksamere Schutz.

**Protokollierung:** Jede verwendete Referenz wird mit ID, Code, Dateiname, Prüfkörper-ID, absolutem Pfad und SHA-256 im Ergebnis gespeichert. **[I+T]**

### 10.3 App-Referenzspeicher (`desktop/reference_store.py`)

- **Aktive Klassen:** Die Klassen werden immer aus `klassifikationsregeln.json` abgeleitet; eine abweichend gespeicherte Auswahl in `aktive_hauptversagensarten.json` wird als `configurationOutOfSync` gemeldet, aber nicht verwendet. **[I+T]**
- **Import:** Mehrfachauswahl über Dateidialog; Abfrage von Prüfkörper-ID, optionaler Beschreibung und Bestätigungsstatus; Prüfung von Format und Lesbarkeit; Kopie nach `<Klassenordner>/<Prüfkörper-ID>/`; Namenskonflikte mit Suffix. **[I+T]**
- **SHA-256-Duplikaterkennung:** Bilder, deren Hash bereits registriert oder als unregistrierte Datei vorhanden ist, werden nicht erneut kopiert und als Duplikat gemeldet. Im Katalog werden hashgleiche Einträge mit `duplicateOf` gekennzeichnet. **[I+T]**
- **Bestätigungsstatus:** `confirmed`, `confirmedBy` (fester Text „App-Benutzerin“), `confirmedAt`. Bereits vorhandene, nicht registrierte Bilder werden als unbestätigt angezeigt und erst durch ausdrückliches Speichern in das Manifest übernommen. **[I+T]**
- **Änderungsschutz:** Änderungen an einem bestätigten Eintrag (Klasse, Prüfkörper-ID, Beschreibung, Aufhebung der Bestätigung) erfordern eine zusätzliche Bestätigung. **[I+T]**
- **Löschschutz:** Es gibt keine Löschfunktion; Quellbilder werden nie verändert. **[I]**
- **Aktueller Bestand:** Ein Manifest `Datenbank/Referenzbilder_neu/referenzbilder_manifest.json` existiert noch nicht. In den Klassenordnern liegen 28 bislang unregistrierte Bilddateien (AA 5, GSF 3, KA 2, LSF 4, RiSch 1, SA 9, Spl HB 2, sp HB 2). Bilder in Ordnern nicht aktiver Klassen (AA, HB) werden in der Oberfläche nicht angezeigt.

**[E]** Wird die Klasse eines Eintrags geändert, bleibt die Datei im Ordner der ursprünglichen Klasse; nur die Metadaten ändern sich. Die Bestätigung wird keiner identifizierten Person zugeordnet. Änderungen werden ohne Versionshistorie überschrieben (nur `updatedAt`).

### 10.4 Schutz vor Datenlecks

| Risiko | Schutz | Nachweis |
|---|---|---|
| Referenzbilder in Variante 1 | Manifest wird nicht geöffnet | **[I+T]** |
| Referenz desselben Prüfkörpers | Namens- und Hashausschluss | **[I+T]** |
| Testbild ist Kopie eines Referenz- oder Kontextbilds | Pfad- und SHA-256-Abgleich (nur Standardmodus) | **[I+T]** |
| Kontextbilder als Beispiel | nur `reference_image` mit Freigabe | **[I+T]** |
| Klassenlabel einer Referenz verrät Testklasse | nicht anwendbar (Referenzen sind andere Prüfkörper) | – |
| App-Referenzspeicher als Input | nicht angebunden | Code |

## 11. Ergebnisse, Expertenreview und Export

### 11.1 Originale KI-Ergebnisse

Jede Analyse erzeugt eine neue Datei `Ergebnisse/<PK>_<provider>_<model>_<UTC-Zeit>_<runId>.json`; vorhandene Dateien werden nicht überschrieben (eindeutige Run-ID). Der Aufbau (Schema 2.0) ist in Anhang B beschrieben und enthält unter anderem: Run-ID, Projekt, Prüfkörper, Zeitstempel, Ausführungsmodus, Provider, angefragte und gemeldete Modellkennung, Laufzeit in Sekunden (vom Start von `run_analysis` bis vor dem Schreiben, also inklusive Bildaufbereitung), Provider-Metadaten (Tokens, Ollama-Zeiten, Request-ID), Einstellungen, Übertragungs- und Kostenkennzeichen, Bildnamen mit SHA-256, Klassifikationsbasis mit Prompt- und Regelinformation, das validierte Ergebnis (Status, primärer Kandidat, sekundäre Merkmale, Beobachtungen, Unsicherheiten, fehlende Ansichten) und feste wissenschaftliche Hinweise. **[I+T]**

Die **rohe Modellantwort wird nicht gespeichert**, nur die validierte und angereicherte Fassung. Bei ungültigen Antworten wird keine Ergebnisdatei geschrieben; der Fehler erscheint nur am Auftrag bzw. in der Batchdatei. **[E]**

### 11.2 Darstellung in der Oberfläche

Die Ergebniskarte (`RunCard`, [frontend/src/App.tsx:25-35](../frontend/src/App.tsx#L25-L35)) trennt sichtbar drei Bereiche: „KI-Beobachtungen“ (inkl. unsicherer Beobachtungen), „Unverbindliche KI-Kandidaten“ (Hauptkandidat, weitere Merkmale mit Konfidenz, aufklappbare bildbasierte Begründungen) und „Fachliche Bewertung“. Darunter stehen Promptversion, Zahl der Referenzen, Tokens und gemeldete Kosten. Im Register Ergebnisse werden zusätzlich die im Lauf verwendeten Prüfkörperbilder angezeigt, sofern sie im Projekt noch vorhanden sind. **[I]** (Frontend nicht automatisiert getestet)

### 11.3 Expertenreview

Die fachliche Bewertung wird je Lauf getrennt in `Bewertungen/<runId>.json` gespeichert ([desktop/bridge.py:1007-1038](../desktop/bridge.py#L1007-L1038)):

- `verdict`: `unreviewed` (Nicht bewertet), `confirmed` (Bestätigt), `partially_correct` (Teilweise richtig), `incorrect` (Falsch), `unclear` (Kein eindeutiges Primärversagen),
- `primaryCode`: fachliche Hauptklasse; darf leer bleiben („Kein eindeutiges Primärversagen“),
- `secondaryCodes`: weitere fachlich bestätigte Merkmale,
- `note`: fachliche Begründung,
- Verweis auf die Ergebnisdatei und Zeitstempel.

Die KI-Ergebnisdatei wird dabei nicht verändert; Speichern, Wiedereinlesen und Export einer Bewertung prüft der Test `test_historical_llava_result_remains_readable_and_exportable`, die getrennte Ablage des Altformats `test_correction_is_stored_separately_from_ai_result`. **[I+T]** Die Oberfläche bietet die zwölf aktiven Merkmale an; der Server akzeptiert alle 29 Codes. Es besteht keine Pflicht, sp HB oder Spl HB zu vergeben.

**[E]** Eine erneute Bewertung überschreibt die vorherige Bewertungsdatei ohne Versionshistorie. Die bewertende Person wird nicht erfasst. Das Altformat `Korrekturen/<PK>.json` (`save_correction`) wird von der aktuellen Oberfläche nicht mehr verwendet.

### 11.4 Vergleich mehrerer Modelle

Im Register Bildanalyse lassen sich bis zu drei Läufe eines Prüfkörpers nebeneinander anzeigen und getrennt bewerten; beim Öffnen sind die drei neuesten Läufe vorausgewählt. Im Register Evaluation werden nur Läufe mit Evaluationsvariante zum Vergleich angeboten. Eine automatische Auswertung der Übereinstimmung zwischen Modellen ist nicht implementiert. **[I]**

### 11.5 Export

Das Register Export ruft `export_selected_runs` auf ([desktop/bridge.py:1119-1181](../desktop/bridge.py#L1119-L1181)) und erzeugt für die ausgewählten Läufe zwei neue Dateien mit Zeitstempel (Mikrosekunden) unter `Export/`:

- **JSON:** Projekt, Exportzeit und je Lauf das vollständige KI-Dokument, die Bewertung und der Quellpfad.
- **CSV** (Trennzeichen Semikolon, UTF-8 mit BOM): Projekt, Prüfkörper, Analysedatum, Run-ID, Evaluationsvariante, Provider, Modell, Prüfkörperbilder, Referenzbildanzahl, Referenzbilder, Promptversion, KI-sichtbare Merkmale, KI-Primärvorschlag, KI-Sekundärvorschläge, bestätigtes Primärversagen, bestätigte weitere Merkmale, fachliche Begründung, Prüfstatus, Laufzeit, Fehlermeldung.

**[I+T]** (Test `test_historical_llava_result_remains_readable_and_exportable`: JSON-Export mit getrennter KI-Ausgabe und Bewertung, neuer Dateiname). **[E]** Die Spalte „Fehlermeldung“ bleibt immer leer, da fehlgeschlagene Läufe keine Ergebnisdatei besitzen. Prompt-Hash und Bild-Hashes sind nur im JSON, nicht in dieser CSV enthalten. Die Bridge enthält weitere, von der Oberfläche nicht genutzte Exportfunktionen; `export_overview` überschreibt bei jedem Aufruf dieselbe Datei.

## 12. Qualitätssicherung, Tests, Build und Installation

### 12.1 Automatisierte Tests

| Modul | Testfälle | Schwerpunkte |
|---|---|---|
| `tests/test_ki_bildanalyse.py` | 33 | Schlüsselauflösung, Katalog und Regelversion 1.2, Referenzauswahl, Variante 1, Promptinhalt, Validierung und Kausalitätsfilter, Live-Sperren, Mock-Pipeline, Mistral-Payload, Teilanfragen und 422-Fehler, Ollama-Parser, Zeitüberschreitung, Leakage-Erkennung |
| `tests/test_desktop_bridge.py` | 20 | Modellauswahl, Bildauswahl, Nummernfotos, Variante-1-Kontrollübersicht, Duplikatsperren, Batch-Freigabe und -Abbruch, LLaVA-Altdaten |
| `tests/test_evaluation.py` | 8 | Ground Truth, Holdout, Laufplan, Variantentrennung, Cloud-Sperre, Variante D, Metriken |
| `tests/test_reference_store.py` | 4 | aktive Klassen, Import und Hash-Duplikate, unregistrierte Bilder, Änderungsschutz |

Framework: `unittest` aus der Standardbibliothek (auch mit `pytest` ausführbar). Alle Modellaufrufe werden durch Mock-Antworten, gefälschte `opener`-Funktionen, gefälschte Clients oder gepatchte `run_analysis`-Funktionen ersetzt; die Tests verwenden temporäre Verzeichnisse.

**Aktueller Testlauf** (26.10.2026, Python 3.12.9, macOS):

```text
python3 -m unittest discover -s tests -v
Ran 65 tests in 23.233s
OK
```

Der Lauf hat keine Dateien im Projekt hinterlassen (Vergleich des Inhalts von `Pruefdaten/Neue_Analysen` vor und nach dem Lauf). In diesem Ordner liegen jedoch sieben leere temporäre Ordner (`tmp…`) früherer Testläufe, die offenbar nicht vollständig entfernt wurden.

**Nicht automatisiert getestet:** React-Oberfläche, pywebview-Dateidialoge (Bild- und Referenzimport über Dialog), echte Provider-Schnittstellen, Inhalt der Export-CSV, `bildzuordnung_neue_analysen.py`, Build und Signatur.

### 12.2 Frontend-Build

`npm --prefix frontend run build` führt `tsc -b && vite build` aus; die Ausgabe liegt in `frontend/dist/` (zuletzt am 26.10.2026 12:22 erzeugt). Die TypeScript-Prüfung ohne Ausgabe lief im aktuellen Prüflauf fehlerfrei. Installierte Versionen: React 19.3.1, Vite 8.3.5, TypeScript 7.0.2, Node.js 26.7.0. **[E]** `package.json` verlangt für alle Abhängigkeiten `latest`; reproduzierbar ist der Build nur über `package-lock.json` und `npm ci`, das `build_macos.sh` verwendet.

### 12.3 macOS-Build und Codesign

`build_macos.sh` installiert die Frontend-Abhängigkeiten mit `npm ci`, baut das Frontend, prüft die Pflichtressourcen, entfernt AppleDouble-Dateien und erweiterte Attribute, baut mit PyInstaller 6.16.0 nach `KI-Bildanalyse.spec` und signiert das Bundle **ad hoc** (`codesign --force --deep --sign -`) in einem temporären Verzeichnis außerhalb des synchronisierten Desktops; dort wird die Signatur streng geprüft. Zielplattform ist macOS ab 12.0 (Info.plist), Architektur arm64.

Prüfung im aktuellen Analyselauf:

- `/Applications/KI-Bildanalyse.app`: „valid on disk“, „satisfies its Designated Requirement“; Signatur ad hoc, ohne Team-ID; Version 1.0.0. Die gebündelten Regeldateien sind hashgleich mit dem Projektstand.
- `dist/KI-Bildanalyse.app` im Desktop-Projekt: strenge Prüfung schlägt wegen des Attributs `com.apple.FinderInfo` fehl. Das entspricht dem im Build-Skript dokumentierten Verhalten des Dateisynchronisationsdienstes.

**[E]** Es gibt keine Signatur mit Developer-ID und keine Notarisierung; auf anderen Rechnern ist mit Gatekeeper-Warnungen zu rechnen. Das Installationsverfahren (Kopie nach `/Applications` ohne erweiterte Attribute) ist im Build-Skript erwähnt, aber nicht als Skript vorhanden. Im Ordner `dist/` liegen zahlreiche Duplikate („KI-Bildanalyse 2.app“ bis „14.app“, ältere „EKI-Bildanalyse“-Bundles).

### 12.4 Installation (Entwicklung)

```bash
python3 -m pip install -r requirements-app.txt     # openai>=2.0.0, Pillow>=10.0.0, pywebview>=6.1
npm --prefix frontend install
npm --prefix frontend run build
python3 app.py
```

Für Qwen3-VL muss eine lokale Ollama-Instanz unter `localhost:11434` mit dem Modell `qwen3-vl:4b` laufen. **[E]** Python-Abhängigkeiten sind nur mit Mindestversionen angegeben; das App-Bundle enthält Module für CPython 3.13, die Entwicklungsumgebung nutzt 3.12.9.

## 13. Datenschutz und Sicherheit

### 13.1 Lokale gegenüber cloudbasierter Analyse

Bei Qwen3-VL verlassen keine Bilder den Rechner; der Endpunkt ist fest auf `localhost` begrenzt. Bei OpenAI und Mistral werden die ausgewählten Prüfkörperbilder (verkleinert) und im Standardmodus zusätzlich bis zu sieben Referenzbilder an die Anbieter übertragen. Jedes Ergebnis kennzeichnet `images_transmitted_to_cloud` sowie den Anbieter.

### 13.2 Technisch nachgewiesene Schutzmechanismen

| Mechanismus | Nachweis |
|---|---|
| Kostenbestätigung vor jedem Cloud-Aufruf (Einzelanalyse) bzw. gemeinsame Freigabe mit Anzahl der Cloud-Anfragen (Batch); in der CLI `--live --confirm-paid-request` | **[I+T]** |
| feste Endpunkte (Ollama lokal, Mistral offizielle HTTPS-API) | **[I+T]** |
| Schlüssel nur aus Umgebung/Schlüsselbund, nie in Dateien, `repr` ausgeblendet | **[I+T]** |
| geschwärzte Fehlermeldungen (Mistral) bzw. schlüsselfreie Fehlermeldungen (OpenAI) | **[I+T]** |
| keine Dateinamen, Prüfkörpernummern oder Protokolldaten im Prompt | **[I+T]** |
| verkleinerte JPEG-Kopien im Arbeitsspeicher, Originale unverändert | **[I+T]** |
| OpenAI `store: false` | Code |
| SHA-256 aller verwendeten Bilder und Referenzen im Ergebnis | **[I+T]** |
| Ausschluss von `Pruefdaten/`, `Output/`, Bildern, Ground Truth, `.env`, Schlüsseldateien aus Git (`.gitignore`) | Konfiguration |

### 13.3 Hinweise und Risiken

- Der OpenAI-Schlüssel wird aus dem Schlüsselbunddienst `KI-Bemessungstool OpenAI` gelesen, der nach Name und Codekommentar aus Use Case 1 stammt. Es handelt sich um eine gemeinsame Zugangsberechtigung, nicht um eine Datenverbindung zwischen den Anwendungen.
- Für Mistral wird kein Parameter zur Unterbindung einer Speicherung gesetzt; die Speicher- und Trainingsbedingungen der Anbieter wurden im Rahmen dieser Analyse nicht geprüft.
- Ergebnis-, Bewertungs- und Batchdateien sind unverschlüsselt und nicht schreibgeschützt; ihre Integrität ist nur durch Zeitstempel und, bei Ergebnissen, durch die gespeicherten Hashes der Eingaben nachvollziehbar. Ein Hash oder eine Signatur der Ergebnisdatei selbst wird nicht erzeugt.
- Absolute Pfade (inklusive Benutzername) stehen in Ergebnis- und Exportdateien (Referenzpfade, Quellpfade).

### 13.4 Empfehlungen (nicht implementiert)

Vor Cloud-Läufen die Nutzungsbedingungen der Anbieter hinsichtlich Speicherung und Training für Forschungsdaten dokumentieren; Freigabeprozess der Forschungseinrichtung für die Übertragung von Versuchsbildern klären; Ergebnisdateien nach Abschluss eines Evaluationslaufs mit Prüfsummen archivieren; Bewertungen versionieren; absolute Pfade in Exporten optional anonymisieren.

## 14. Diskussion und Grenzen

**Eingeschränkte Bildperspektiven.** Es liegen nur Draufsichten beider Breitseiten vor. Faserneigung in Dickenrichtung, Astverläufe auf der Schmalseite, Jahrringgrenzen im Querschnitt und Risstiefen sind nicht beurteilbar. Die Regeln tragen dem Rechnung, indem sie keine zusätzlichen Perspektiven voraussetzen und „nicht beurteilbar“ verlangen; die Grenze selbst bleibt bestehen. Besonders betroffen sind die Abgrenzung KA/SA (Kantenberührung), GSF/LSF (Ausdehnung) und RiSch (Jahrringbindung).

**Fehlende Querschnittsansichten.** Für RiSch und die Bruchmorphologie (sp HB/Spl HB) wären Stirnholz- und Bruchflächenansichten aussagekräftiger. Ohne sie beschreibt die KI im besten Fall ein „ringschälenartiges“ Merkmal.

**Begrenzte Referenzbilder.** 18 bestätigte Hauptreferenzen für die ursprünglichen acht Klassen, 1–3 je Klasse, keine für LR; alle stammen aus den Serien Z-A bis Z-G. Die Referenzen bilden die Variabilität von Holzart, Feuchte, Beleuchtung und Probengeometrie nicht ab.

**Visuelle Mehrdeutigkeit von Holzmerkmalen.** Jahrringzeichnung und Faserrichtung, Schatten und Risse, Bruchkanten und Faserverlauf sind auf Fotos leicht zu verwechseln; die Regeln benennen diese Verwechslungen ausdrücklich.

**Unsicherheit bei Versagensursachen.** Die Ursache eines Bruchs ist aus Nachbruchfotos allein nicht nachweisbar. Die Software trennt deshalb Merkmal, Bruchbild und Ursache konsequent. Auch die fachliche Bewertung im Review beruht auf denselben Fotos und gegebenenfalls zusätzlicher Erfahrung; sie ist eine fachliche Einschätzung, keine Messung.

**Modellhalluzinationen.** Modelle können nicht vorhandene Merkmale beschreiben oder Merkmale der Referenzbilder auf das Testbild übertragen. Das Schema verhindert ungültige Codes, nicht aber falsche Beobachtungen. Die historische LLaVA-Antwort (Primärklasse KB mit Konfidenz 0,0 bei Beleg „Klasse: Spl HB“) ist ein dokumentiertes Beispiel einer formal gültigen, inhaltlich widersprüchlichen Antwort.

**Unterschiede zwischen Cloud- und lokalen Modellen.** Modellgröße, Bildauflösung, Laufzeit und Parameter unterscheiden sich (8.6). Ein Vergleich misst daher immer die Kombination aus Modell und Konfiguration.

**Begrenzte Reproduzierbarkeit generativer Modelle.** Auch mit Temperatur 0 sind Ausgaben nicht garantiert identisch (Hardware, Modellversionen, serverseitige Änderungen). Bei OpenAI ist die Temperatur nicht gesetzt. Reproduzierbar dokumentiert sind die Eingaben (Hashes), nicht die Ausgaben. Wiederholungsläufe sind daher für eine Aussage über Stabilität erforderlich.

**Verzerrungen durch Klassenbeschreibungen.** Die Reihenfolge der Klassen, die Ausführlichkeit einzelner Regeln und die Beispiele für Mehrfachklassifikation (KA mit LSF, RiSch mit Spl HB) können die Vorschläge beeinflussen. Im Standardmodus kommt die Angabe `confirmed_reference_count` hinzu. Diese Effekte sind nicht untersucht.

**Fachliche Validierung der Klassifikationsregeln.** Nur KA ist „vorläufig bestätigt“; vier Klassen sind „bedingt geeignet“, drei „fachlich offen“. Schwellenwerte (Längen, Winkel, Splitteranteile) fehlen bewusst. Die Ergebnisse jeder Evaluation hängen von diesem Regelstand ab.

## 15. Ausblick

- **Erweiterung der Klassen:** Weitere Klassen werden aktiv, sobald sie in `klassifikationsregeln.json` mit allen Pflichtfeldern dokumentiert sind; Code und Referenzspeicher übernehmen sie ohne Änderung. Die Definitionen von 21 weiteren Klassen liegen in `versagensarten_erweitert.json` bereits vor, sind aber größtenteils als prüfbedürftig markiert.
- **Zusätzliche bestätigte Referenzbilder:** insbesondere für LR und RiSch; dazu Anbindung des App-Referenzspeichers an die Modellaufrufe mit klar getrennten Datensatzrollen (Referenz vs. Test).
- **Bildqualität und standardisierte Aufnahmen:** feste Kameraposition, Maßstab, Beleuchtung, gegebenenfalls zusätzliche Bruchflächen- und Schmalseitenaufnahmen.
- **Systematische wissenschaftliche Evaluation:** Festlegung der Varianten 2a, 2b und 3, unabhängiger Testdatensatz, Mehrfachläufe, Multi-Label-Metriken, Interrater-Vergleich.
- **Erweiterte Analyseverfahren:** technisch getrennte Zweiphasigkeit (wie Variante D) in der App, bildweise Belege, Kalibrierung von Konfidenzen.
- **Mobile Anwendung:** denkbar für die Aufnahme am Prüfstand; derzeit nicht vorbereitet.
- **Übergabe an Use Case 3:** Die vorhandenen CSV- und JSON-Exporte enthalten KI-Vorschlag und fachliche Bewertung getrennt und könnten als Austauschformat für die nachgelagerte statistische Auswertung und Prüfberichtserstellung dienen. Ein festgelegtes Schnittstellenformat existiert noch nicht. Eine Verbindung zu Use Case 1 ist nicht vorgesehen.

## 16. Zusammenfassung des Entwicklungsstands

| Bereich | Stand |
|---|---|
| Desktop-App, Projekt- und Bildverwaltung, Nummernfoto-Ausschluss, Bildauswahl | implementiert, Kernlogik getestet |
| Acht Klassen, Regelversion 1.2, Prompt 3.1 | implementiert und getestet; noch kein echter Modelllauf |
| Anbindung Qwen3-VL/Ollama, OpenAI, Mistral | implementiert, mit Fakes getestet; historische Live-Läufe mit älteren Ständen |
| Variante 1 | implementiert und getestet; kein echter Lauf |
| Varianten 2a, 2b, 3 | nicht definiert |
| Evaluation A–D (CLI) | implementiert und getestet; nicht ausgeführt |
| Referenzbilddatenbank | kuratierter Bestand (26) in Nutzung; App-Speicher implementiert, noch ohne Manifest, nicht an Modelle angebunden |
| Expertenreview, Export | implementiert und getestet (Speichern, Wiedereinlesen, JSON-Export) |
| Build, Signatur | ad hoc signiertes Bundle installiert und gültig; keine Notarisierung |
| Fachliche Validierung der Klassifikationsgüte | **nicht vorhanden** |

---

## Anhang A: Exemplarischer Prompt (Variante 1, ohne Bilddaten, ohne Modellaufruf)

Der vollständige Prompt liegt maschinenlesbar in [Datenbank/klassifikationsprompt_v1.3_beispiel.json](../Datenbank/klassifikationsprompt_v1.3_beispiel.json) vor und ist identisch mit dem Ergebnis von `_analysis_instructions()` für die zwölf aktiven Merkmale. Gekürzte Darstellung der Nachrichtenfolge für Qwen3-VL über Ollama (Bilder durch Platzhalter ersetzt, die Klassenliste nach der ersten Klasse gekürzt):

```json
[
  {"role": "system",
   "content": "Du sichtest ausschliesslich sichtbare Merkmale von Holz-Zugproben. Du gibst keine nachgewiesene Bruchursache aus und haeltst dich strikt an das JSON-Schema."},
  {"role": "user",
   "content": {
     "task": "Zuerst neutrale Sichtung, danach unverbindliche Klassifikationskandidaten einer unbekannten Holz-Zugprobe",
     "rules": [
       "Arbeite in zwei strikt getrennten Phasen: (1) sichtbare Beobachtungen beschreiben, (2) erst danach mögliche Klassencodes vorschlagen.",
       "Nenne in Phase 1 Risse und Rissverläufe, Splitter/Auffaserung, mögliche Jahrringtrennungen, Äste und deren Position, Faserrichtung, Bruchflächen und weitere tatsächlich sichtbare Auffälligkeiten.",
       "Untersuche den gesamten sichtbaren Prüfkörper und dokumentiere relevante Merkmale innerhalb und außerhalb der Bruchzone.",
       "Kennzeichne unsichere Beobachtungen ausdrücklich als unsicher. Unterscheide: nicht sichtbar = in den Bildern nicht erkennbar; nicht beurteilbar = Bildqualität, Ausschnitt oder Perspektive erlauben keine verlässliche Entscheidung.",
       "Verwende ausschliesslich die bereitgestellten 8 Klassencodes oder UNKLAR.",
       "Alle bereitgestellten Klassen sind gleichberechtigt. Erzwinge insbesondere weder sp HB noch Spl HB.",
       "Mehrere passende Klassen sind zulässig. Markiere bei ausreichender Evidenz einen primären Klassifikationskandidaten und weitere passende Klassen als sekundäre Merkmale; dies ist keine bestätigte Hauptversagensursache.",
       "Begruende jede Klasse ausschliesslich mit konkret sichtbaren Bildmerkmalen.",
       "Behaupte niemals, ein Holzmerkmal sei die nachgewiesene Versagens- oder Bruchursache.",
       "Gib UNKLAR aus, wenn Bildausschnitt, Perspektive, Schaerfe oder sichtbare Merkmale nicht ausreichen.",
       "Erfinde keine Winkel, Längen, Anzahlen oder sonstigen Entscheidungsschwellen.",
       "Beurteile ausschließlich die ausgewählten unbekannten Prüfkörperbilder und die bereitgestellten Klassenbeschreibungen.",
       "Halte die Ausgabe knapp: hoechstens 3 Belege je Merkmal, 4 sekundaere Merkmale, 6 Beobachtungen sowie je 4 Unsicherheiten und fehlende Ansichten.",
       "Bei Status UNKLAR muss primary_code UNKLAR sein; primary_evidence und secondary_features muessen dann leer sein."
     ],
     "classes": [
       {"code": "GSF", "name": "Globale Schrägfaser",
        "definition": "Globale Schrägfaser liegt vor, wenn der erkennbare Faserverlauf über einen größeren sichtbaren Abschnitt des Prüfkörpers gleichgerichtet von dessen Längsachse abweicht.",
        "positive_image_features": ["…drei Einträge…"],
        "distinguishing_features": ["…vier Einträge…"],
        "uncertainty_rules": {"do_not_assign_when": ["…"], "unclear_when": ["…"], "visual_limitations": ["…"]}},
       "… LSF, KA, SA, sp HB, Spl HB, RiSch, LR in gleicher Struktur …"
     ]
   }},
  {"role": "user", "content": "Ab hier folgen ausschliesslich ausgewaehlte Bilder des unbekannten Testpruefkoerpers."},
  {"role": "user", "content": "Unbekanntes Testbild 1.", "images": ["<JPEG Base64, max. 768 px>"]},
  {"role": "user", "content": "Unbekanntes Testbild 2.", "images": ["<JPEG Base64, max. 768 px>"]},
  {"role": "user", "content": "Erzeuge jetzt ausschliesslich die knappe JSON-Klassifikation aller bereitgestellten Testbilder. Beschreibe keine Referenzbilder."}
]
```

Im tatsächlichen Aufruf wird der Benutzerprompt als kompakter JSON-**String** übertragen; hier ist er zur Lesbarkeit als Objekt dargestellt. Zusätzlich werden `format` (JSON-Schema mit den acht Codes und `UNKLAR`), `stream: false`, `think: false` und `options: {temperature: 0, num_predict: 4096, num_ctx: 24576}` gesendet.

## Anhang B: Ergebnisschema 2.0 (Auszug)

```text
schema_version            "2.0"
run_id, project_id, specimen_id, analysis_id, created_at
execution
  mode                    "local" | "live" | "mock"
  provider, model, requested_model, actual_model
  model_verification      Eintrag aus der Verifikationsdatei (Cloud) oder null
  duration_seconds
  provider_metadata       Tokens, Zeiten, Request-ID, Optionen, ggf. image_batches (Mistral)
  settings                evaluation_variant, max_references_per_class, ollama_*
  paid_api_request_performed, images_transmitted_to_openai|mistral|cloud
inputs[]                  {filename, sha256}
image_selection           {included[], excluded[]}
classification_basis
  prompt_version, prompt_hash_sha256, evaluation_variant
  allowed_class_count, allowed_class_codes[]
  classification_rules_version, active_class_set_version, active_class_descriptions_sha256
  reference_database_accessed
  confirmed_main_reference_count_available, confirmed_main_reference_count_used
  protocol_failure_cause_transmitted, ground_truth_transmitted,
  previous_results_transmitted, expert_assessments_transmitted   (jeweils false)
  only_confirmed_main_references_used
  references[]            {id, code, filename, specimen_id, path, sha256}
classes_without_confirmed_references[]
result
  status                  "KLASSIFIZIERT" | "UNKLAR"
  primary_feature         {code, name, feature_type, confidence, image_based_reasoning[],
                           has_confirmed_reference_image, confirmed_reference_count,
                           proven_failure_cause: false, causality_note} | null
  secondary_features[]    gleiche Struktur
  visible_observations[], uncertainty_reasons[], missing_views[]
scientific_limits[]
```

## Anhang C: Verzeichnis der zitierten Projektdateien

[app.py](../app.py) · [desktop/main.py](../desktop/main.py) · [desktop/bridge.py](../desktop/bridge.py) · [desktop/reference_store.py](../desktop/reference_store.py) · [Tool/ki_bildanalyse.py](../Tool/ki_bildanalyse.py) · [Tool/evaluation.py](../Tool/evaluation.py) · [Tool/bildzuordnung_neue_analysen.py](../Tool/bildzuordnung_neue_analysen.py) · [Tool/openai_config.py](../Tool/openai_config.py) · [Tool/mistral_config.py](../Tool/mistral_config.py) · [Tool/model_verification.json](../Tool/model_verification.json) · [Tool/mistral_model_verification.json](../Tool/mistral_model_verification.json) · [Tool/mock_response.json](../Tool/mock_response.json) · [Tool/README.md](../Tool/README.md) · [Datenbank/klassifikationsregeln.json](../Datenbank/klassifikationsregeln.json) · [Datenbank/versagensarten_erweitert.json](../Datenbank/versagensarten_erweitert.json) · [Datenbank/versagensarten.json](../Datenbank/versagensarten.json) · [Datenbank/klassifikationsprompt_v1.3_beispiel.json](../Datenbank/klassifikationsprompt_v1.3_beispiel.json) · [Datenbank/Pruefung_Versagensarten.md](../Datenbank/Pruefung_Versagensarten.md) · [Datenbank/Referenzbilder/referenzbilder_manifest.json](../Datenbank/Referenzbilder/referenzbilder_manifest.json) · [Datenbank/Referenzbilder_neu/README.md](../Datenbank/Referenzbilder_neu/README.md) · [Evaluation/README.md](../Evaluation/README.md) · [frontend/src/App.tsx](../frontend/src/App.tsx) · [KI-Bildanalyse.spec](../KI-Bildanalyse.spec) · [build_macos.sh](../build_macos.sh) · [APP_README.md](../APP_README.md) · [tests/](../tests/)

Externe wissenschaftliche Quellen werden in diesem Bericht nicht zitiert. Die in `klassifikationsregeln.json` genannten Literatur- und Normquellen sind Bestandteil der Projektdatei; ihr Inhalt wurde im Rahmen dieser Dokumentation nicht erneut geprüft.

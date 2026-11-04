# Benutzerhandbuch – KI-Bildanalyse

**Version der Anwendung:** 1.0.0 · **Stand des Handbuchs:** 02.11.2026
**Für wen:** Anwenderinnen und Anwender, die Fotos zerstörter Holz-Zugprüfkörper mit KI-Unterstützung sichten und fachlich bewerten.

Alle Schaltflächen und Register sind so benannt, wie sie in der Oberfläche erscheinen (geprüft anhand von `frontend/src/App.tsx`).

---

## Das Wichtigste vorab

- Die KI **beschreibt sichtbare Merkmale** und **schlägt Klassen vor**. Sie stellt **keine Versagensursache** fest.
- Die **fachliche Hauptversagensursache bestimmen Sie** im Bereich „Fachliche Bewertung“. Ihre Bewertung wird getrennt gespeichert; die KI-Antwort bleibt unverändert.
- Sie müssen weder „sp HB“ noch „Spl HB“ vergeben. „Kein eindeutiges Primärversagen“ ist eine zulässige Bewertung.
- **Qwen3-VL 4B** arbeitet **lokal** auf dem Mac. **OpenAI** und **Mistral** sind **Cloud-Dienste**: Die ausgewählten Bilder werden übertragen und es können **Kosten** entstehen. Die App fragt vorher immer nach.
- Ihre Originalbilder werden nie verändert, verschoben oder gelöscht.

Aktive Referenzmerkmale (elf): GSF Globale Schrägfaser · LSF Lokale Schrägfaser · KA Kantenast · SA Seitenast · sp HB Spröder Holzbruch · Spl HB Splitterholzbruch · RiSch Ringschäle · LR Längsriss · Bu Buchs · AA Astansammlung · SSA Schmalseitenast.

---

## 1. Anwendung öffnen

1. **Voraussetzung für Qwen3-VL:** Die Ollama-App muss laufen (Symbol in der Menüleiste) und das Modell `qwen3-vl:4b-instruct` muss installiert sein. Die KI-Bildanalyse startet Ollama nicht selbst.
2. Öffnen Sie **KI-Bildanalyse** aus dem Ordner „Programme“.
3. Es erscheint für einige Sekunden ein Startbildschirm, danach die Startseite.

> Hinweis: Die App ist nicht mit einem Apple-Entwicklerzertifikat signiert. Auf einem anderen Mac kann macOS beim ersten Öffnen warnen; öffnen Sie die App dann per Rechtsklick → „Öffnen“.

## 2. Projekt auswählen oder anlegen

Auf der Startseite stehen vier Schaltflächen zur Verfügung:

| Schaltfläche | Wirkung |
|---|---|
| **Beispielprojekt Test_001 öffnen** | öffnet die vorhandene Testserie Z-30 |
| **Bestehendes Projekt öffnen** | Ordnerauswahl. Wählen Sie den Projektordner; enthält er einen Unterordner `Sortiert`, wird dieser automatisch verwendet. |
| **Neues Projekt erstellen** | fragt nach einem Projektnamen und legt `Pruefdaten/Neue_Analysen/<Name>/Sortiert` an |
| **Einzelne Bilder hochladen** | legt ein neues Projekt an, fragt nach einer Prüfkörpernummer und öffnet den Dateidialog für die Bilder |

Ein Projekt besteht aus einem Ordner je Prüfkörper (z. B. `Z-30-12`). Nummernfotos haben im Dateinamen das Präfix `Nummer_` – so erstellt sie das Sortierwerkzeug für neue Fotoserien.

Oben rechts wechseln Sie mit **Projekt wechseln** jederzeit zurück zur Startseite.

## 3. Prüfkörper auswählen

Die Oberfläche hat fünf Register: **Bildanalyse**, **Ergebnisse**, **Evaluation**, **Referenzdatenbank**, **Export**.

Im Register **Bildanalyse** steht links die Liste **PRÜFKÖRPER**. Je Eintrag sehen Sie die Zahl der Fotos, der analysierbaren Fotos und der bisherigen Läufe.

- Klick auf den Namen öffnet den Prüfkörper.
- Die Checkbox vor dem Namen wählt den Prüfkörper für eine **Serienauswertung** (Batch) aus; **Alle** / **Keine** wählt alle bzw. keinen.
- Mit **＋** legen Sie einen neuen Prüfkörper an (eindeutige Nummer erforderlich).

## 4. Bilder kontrollieren

Nach dem Öffnen eines Prüfkörpers zeigt die Galerie **alle** Fotos. Jede Kachel ist gekennzeichnet:

| Kennzeichnung | Bedeutung |
|---|---|
| **ANALYSEBILD** | wird an das Modell übertragen |
| **NICHT AUSGEWÄHLT** | wird nicht übertragen |
| **IDENTIFIKATION · NICHT ANALYSIERT** | Nummernfoto; kann nicht ausgewählt werden und wird nie übertragen |

- Beim ersten Öffnen sind bis zu **drei** über die Serie verteilte Bruchbilder vorausgewählt.
- Klick auf ein Bild oder dessen Checkbox wählt es an oder ab. Die Auswahl wird sofort gespeichert und beim nächsten Öffnen wiederhergestellt.
- **Alle Bruchbilder auswählen** / **Alle abwählen** ändern die Auswahl für alle Bruchbilder.
- **Vergrößern** öffnet die Großansicht; mit ‹ und › blättern Sie, Klick auf den Hintergrund schließt sie.
- **Bilder hinzufügen** kopiert weitere Fotos in den Prüfkörperordner (gleichnamige Dateien erhalten automatisch ein Suffix wie `_2`).

**Empfehlung:** Wählen Sie Bilder beider Breitseiten und, falls vorhanden, eine Detailansicht der Bruchzone. Unscharfe oder stark verdeckte Aufnahmen führen eher zu „UNKLAR“. Achten Sie darauf, dasselbe Foto nicht doppelt zu importieren – die App erkennt doppelte Prüfkörperbilder nicht automatisch.

## 5. KI-Modell auswählen

| Anzeige | Ort | Hinweise |
|---|---|---|
| **Qwen3-VL 4B · lokal** | auf dem Mac (Ollama) | keine Kosten, keine Übertragung; dauert erfahrungsgemäß einige Minuten |
| **OpenAI GPT-6.1 · Cloud** | Cloud | Kosten möglich, Bilder werden übertragen |
| **Mistral Ministral 3B · Cloud** | Cloud | Kosten möglich, Bilder werden übertragen; bei mehreren Bildern mehrere Anfragen |

- **Einzelanalyse:** Modell im Auswahlfeld rechts über der Galerie wählen.
- **Serienauswertung:** Modelle in der Leiste „Serienauswertung“ per Checkbox wählen (mehrere möglich).

## 6. Evaluationsvariante einstellen

Die Evaluationsvariante wird **nicht** im Register Bildanalyse eingestellt.

- Analysen im Register **Bildanalyse** (einzeln oder als Batch) laufen im **Standardmodus**: Neben den Klassenregeln erhält das Modell je Klasse ein bestätigtes Referenzbild aus der kuratierten Referenzdatenbank.
- **Variante 1:** ausgewählte Prüfkörperbilder und elf Merkmalsbeschreibungen; keinerlei Referenz- oder Annotationszugriff.
- **Variante 2:** kleiner, fester Bestand unveränderter Referenzbilder aus `Variante_2`.
- **Variante 2a:** größerer, fester Bestand unveränderter Referenzbilder aus `Variante_2a`; Zielmengen stehen ausschließlich im Manifest.
- **Variante 3:** manuell annotierte Bildkopien aus `Variante_3` mit konkreter Beschreibung und Verweis auf das unveränderte Original aus 2a. Standardmäßig wird nur die annotierte Kopie als Referenzinput verwendet.

Allgemeine Klassenbeschreibungen stammen bei allen Varianten aus denselben Regeln. Annotationen werden nicht automatisch erzeugt; Variante 3 ist kein Training oder Fine-Tuning.

## 7. Evaluation vorbereiten

1. Register **Evaluation** öffnen.
2. Testreihe, Evaluationsvariante, **Prüfkörper** und KI-Modell wählen.
3. Die kurze Variantenbeschreibung und den Bereitschaftsstatus prüfen.
4. Ist eine Variante nicht bereit, **Referenzdatenbank öffnen** wählen und die
   dort genannten fehlenden Merkmale beziehungsweise Dateien ergänzen.

Technische Bestandszahlen, Hash- und Quellenprüfungen, Referenzgedächtnis,
erwartete lokale beziehungsweise Cloud-Aufrufe sowie die Kontrollübersicht sind
weiterhin unter **Technische Details anzeigen** verfügbar. Im normalen Ablauf
prüft die App beim Start automatisch Ordnerstruktur, Variante, Referenzbilder,
Beschreibungen und Kontrollübersicht. Eine veraltete Kontrollübersicht kann daher
nicht unbemerkt wiederverwendet werden.

## 8. Analyse starten

**Einzelanalyse (Standardmodus):** Im Register Bildanalyse **Bildanalyse starten** klicken. Bei Cloud-Modellen erscheint ein Bestätigungsdialog zu Kosten und Bildübertragung; ohne Bestätigung wird nichts gesendet. Während der Analyse zeigt ein Fenster den Status („Modell wird geladen“, „Bilder werden analysiert“, „Ergebnis wird verarbeitet“) und die vergangene Laufzeit.

**Evaluationsvarianten:** Im Register Evaluation **Analyse starten** klicken. Die
App führt zuerst alle technischen Prüfungen aus. Fehlen Referenzen, Dateien,
Beschreibungen oder vorbereitetes Referenzwissen, wird vor dem Modellaufruf mit
einer verständlichen Meldung abgebrochen. Bei Cloud-Modellen folgt erst danach
die ausdrückliche Kosten- und Übertragungsfreigabe.

**Serienauswertung (Batch):**
1. Prüfkörper links per Checkbox und Modelle in der Leiste „Serienauswertung“ wählen.
2. Optional **Plan prüfen**: zeigt die Zahl der Analysen und der Cloud-Anfragen.
3. **Batch starten**. Bei Cloud-Modellen erscheint eine gemeinsame Kosten- und Übertragungsfreigabe mit der Zahl geplanter Cloud-Anfragen.
4. Die Aufträge laufen nacheinander. Je Auftrag sehen Sie Prüfkörper, Modell, Status, Laufzeit und gegebenenfalls die Fehlermeldung.
5. Mit **Aus Anzeige entfernen** blenden Sie einen erfolgreichen, fehlgeschlagenen oder abgebrochenen Auftrag einzeln aus. **Verlauf leeren** blendet alle Einträge mit diesen drei Endzuständen aus. Laufende, wartende und unterbrochene Aufträge bleiben sichtbar.

Das Bereinigen des Batch-Verlaufs löscht ausschließlich Anzeigeeinträge. Die vollständigen Batch-Manifeste, gespeicherten KI-Ergebnisse, Bilddateien, Prüfprotokolle und Expertenbewertungen bleiben erhalten. Das Löschen solcher Analyseartefakte ist eine andere, hier bewusst nicht angebotene Operation.
5. **Wartende abbrechen** bricht noch nicht gestartete Aufträge ab. **Unterbrochene fortsetzen** setzt unterbrochene (z. B. nach Schließen der App) und fehlgeschlagene Aufträge fort; erfolgreiche Aufträge werden nicht wiederholt.

Derselbe Prüfkörper kann nicht gleichzeitig zweimal mit demselben Modell analysiert werden.

## 9. Ergebnisse lesen

Jede Ergebniskarte hat drei klar getrennte Bereiche:

1. **KI-Beobachtungen:** Was das Modell gesehen haben will, darunter „Unsichere Beobachtungen“.
2. **Unverbindliche KI-Kandidaten:** **HAUPTKANDIDAT** (oder „UNKLAR“), **Weitere Merkmale** mit Prozentangabe; unter „Bildbasierte Begründungen“ die Belege.
3. **Fachliche Bewertung:** Ihre Eingabe (Abschnitt 10).

Darunter stehen Promptversion, Zahl der Referenzbilder, Tokens und gemeldete Kosten (in der Regel „nicht gemeldet“).

**So lesen Sie die Angaben richtig:**
- Die Prozentwerte schätzt das Modell selbst. Sie sind **keine geprüfte Wahrscheinlichkeit**.
- Der Hauptkandidat ist ein **Vorschlag**, keine bestätigte Hauptversagensursache.
- „UNKLAR“ ist eine zulässige und oft angemessene Antwort, wenn die Fotos keine sichere Aussage erlauben.
- Prüfen Sie jede Beobachtung am Bild. Modelle können Merkmale beschreiben, die nicht vorhanden sind.

Im Register **Ergebnisse** sehen Sie alle Läufe des Projekts mit Filtern nach Prüfkörper, Modell und vorgeschlagener Versagensart. Klick auf den Prüfkörpernamen zeigt die im Lauf verwendeten Bilder und die Ergebniskarte. **Aktualisieren** lädt die Liste neu. Läufe älterer Modelle (z. B. LLaVA) bleiben lesbar.

## 10. Expertenreview durchführen

Im Bereich **Fachliche Bewertung** jeder Ergebniskarte:

1. **Gesamturteil** wählen: Nicht bewertet · Bestätigt · Teilweise richtig · Falsch · Kein eindeutiges Primärversagen.
2. **Fachliche Hauptklasse** wählen oder „Kein eindeutiges Primärversagen“ belassen.
3. **Weitere Merkmale** im Mehrfachauswahlfeld markieren (Cmd-Taste für mehrere).
4. Optional eine **fachliche Begründung** eintragen.
5. **Bewertung speichern**.

Die Bewertung wird getrennt von der KI-Antwort gespeichert. Speichern Sie erneut, wird Ihre vorherige Bewertung **ersetzt** (es gibt keine Versionsgeschichte). Ihr Name wird nicht gespeichert – vermerken Sie ihn bei Bedarf in der Begründung.

## 11. Ergebnisse vergleichen und exportieren

**Vergleichen:** Unter **Analysehistorie** (Register Bildanalyse) können Sie bis zu **drei** Läufe eines Prüfkörpers per Checkbox nebeneinander anzeigen und jeweils getrennt bewerten. Beim Öffnen sind die drei neuesten Läufe vorausgewählt. Im Register Evaluation werden unter „Variantenläufe vergleichen“ nur Läufe einer Evaluationsvariante angeboten.

**Exportieren:**
1. Register **Export** öffnen.
2. Gewünschte Läufe per Checkbox wählen.
3. **„n Läufe exportieren“** klicken.

Es entstehen zwei **neue** Dateien im Projektordner unter `Output/EKI_Bildanalyse/Projekte/<Projekt>/Export/`: eine CSV-Datei (Semikolon-getrennt, für Excel geeignet) und eine JSON-Datei mit allen Details. Vorhandene Exporte werden nicht überschrieben. Der Pfad wird in der Statuszeile unten angezeigt. KI-Vorschlag und Ihre Bewertung stehen in getrennten Spalten.

## 12. Referenzdatenbank verwalten

Das Register **Referenzdatenbank** liest direkt die Ordner unter
`Datenbank/Evaluationsvarianten`. Je aktives Referenzmerkmal sehen Sie die allgemeine
Kurzbeschreibung, Bestände von Variante 2 und 2a sowie die manuell annotierten
Bilder von Variante 3. Galerien zeigen Dateiname, stabile ID, SHA-256,
Bestätigungsstatus und Beschreibung. Bei Variante 3 können Original aus 2a und
annotierte Ableitung aus 3 nebeneinander geöffnet werden.

1. **Änderungen prüfen** liest die Ordner neu ein und zeigt neue, fehlende oder
   veränderte Dateien, fehlende Texte, ungültige Quellen und Duplikate. Es wird
   noch kein Manifest geschrieben.
2. **Referenzdatenbank aktualisieren** zeigt dieselbe Vorschau und verlangt eine
   ausdrückliche Bestätigung. Erst danach werden Inventare atomar aktualisiert.

Die App löscht, verschiebt, benennt oder verändert dabei kein Bild. Bereits
eingefrorene Testreihen werden nicht umgeschrieben. Annotationen und Texte für
Variante 3 werden ausschließlich manuell erstellt. Bu, AA und SSA erscheinen
gleichwertig mit den bisherigen acht Referenzmerkmalen; Bu bleibt als fachlich
zu prüfen gekennzeichnet. Mit **Ordner im
Finder öffnen** wird nur der vorhandene, fest freigegebene Merkmalsordner
geöffnet; die App erzeugt oder verändert dabei keine Datei.

Im Abschnitt **Fachliche Bewertung** ist die Auswahl **„Primärversagen manuell
auswählen“** freiwillig. Der Standardwert **„Nicht festgelegt“** ist gültig.
Bei einer Auswahl werden Merkmalscode, bewertende Person und Zeitpunkt getrennt
von der unveränderten KI-Ausgabe gespeichert und exportiert. Alle elf Merkmale
können gemeinsam als sichtbare Mehrfachmerkmale bestätigt
werden; daraus folgt keine automatische Ursachen- oder Primärzuordnung.

## 13. Typische Fehler beheben

| Meldung / Situation | Ursache | Abhilfe |
|---|---|---|
| „Lokales Ollama ist unter http://localhost:11434/api/chat nicht erreichbar“ | Ollama läuft nicht | Ollama starten, Analyse erneut starten |
| „Ollama-HTTP-Fehler 404 …“ | Modell `qwen3-vl:4b-instruct` nicht installiert | im Terminal `ollama pull qwen3-vl:4b-instruct` (lädt das Modell herunter) |
| „Zeitüberschreitung: Ollama hat innerhalb von 900 Sekunden nicht geantwortet …“ | Mac ausgelastet oder zu viele Bilder | weniger Bilder auswählen, andere Programme schließen, erneut versuchen |
| „Ollama-Antwort enthaelt kein lesbares strukturiertes JSON …“ | Modell hat ungültig geantwortet | erneut starten; bei Wiederholung weniger Bilder |
| „Modellantwort: unzulaessige Ursachenaussage …“ | Modell hat eine Ursache behauptet | Antwort wird bewusst verworfen; erneut starten |
| „… erfordert eine ausdrückliche Kostenbestätigung“ | Cloud-Dialog abgelehnt | erneut starten und bestätigen, falls Kosten freigegeben sind |
| „Kein OpenAI-Zugang gefunden …“ / „Kein Mistral-Schlüssel …“ | Schlüssel fehlt im Schlüsselbund | Zugang durch die zuständige Person einrichten lassen |
| „Bitte wählen Sie mindestens ein Analysebild aus.“ | keine Auswahl | Bild in der Galerie auswählen |
| „Variante … ist noch nicht bereit“ | Referenzbilder, Dateien oder Beschreibungen fehlen | **Referenzdatenbank öffnen** und die genannten Punkte ergänzen |
| „Referenzwissen ist noch nicht vorbereitet“ | technische Vorbereitung für Variante 2, 2a oder 3 fehlt | **Technische Details anzeigen** öffnen und Referenzwissen vorbereiten |
| „Der erste Pilotlauf ist auf Qwen3-VL über Ollama begrenzt.“ | Variante 1 mit anderem Modell | Variante 1 nur mit Qwen3-VL |
| „Für … läuft mit … bereits eine Einzelanalyse / ein Batch-Auftrag“ | gleicher Auftrag läuft | Abschluss abwarten |
| „Ein Testbild ist inhaltlich identisch mit einem bekannten Referenz- oder Kontextbild …“ | ein Prüfkörperbild ist auch Referenzbild | dieses Bild abwählen |
| „Diese Prüfkörpernummer existiert bereits.“ | doppelte Nummer | andere Nummer verwenden |
| Batch zeigt „Unterbrochen“ | App wurde während des Laufs geschlossen | **Unterbrochene fortsetzen** |
| Falsche Zahl aktiver Merkmale | Konfiguration und App-Build passen nicht zusammen | App neu starten; die Anzeige wird dynamisch aus der zentralen Liste der elf Merkmale erzeugt |
| „Python-Bridge ist noch nicht bereit.“ | App startet noch | kurz warten |

# Projektarbeitsstand KI-Bildanalyse

Stand: 2026-11-04

## Referenzdatenbank

- Varianten 1, 2, 2a und 3 sind technisch vorbereitet.
- Variante 1 ist referenzfrei und greift nicht auf die Variantenordner zu.
- Variante 2: 1–4 Bilder; Ziel 2 für die bisherigen acht, Ziel für Bu/AA/SSA/KB offen.
- Variante 2a: Ziel 20 für die bisherigen acht, Ziel für Bu/AA/SSA/KB offen.
- Variante 3: eigene manuell annotierte Kopien, Zuordnung zum Original aus 2a
  über `source_reference_id`, individuelle Beschreibung in `descriptions.csv`.
- Die Ordner der Varianten 2, 2a und 3 sind fachlich noch zu befüllen und werden
  bis dahin neutral als **NICHT BEREIT** angezeigt.
- Die alte Datenbank `Datenbank/Referenzbilder_neu` bleibt historisch erhalten
  und wird nicht automatisch migriert.

## Merkmale und Bewertung

- Zwölf gleichwertige Referenzmerkmale in fester Reihenfolge: GSF, LSF, KA, SA,
  sp HB, Spl HB, RiSch, LR, Bu, AA, SSA, KB.
- Bu, AA, SSA und KB besitzen eigene Ordner in Variante 2, 2a und 3 und stehen in
  `active_class_codes`. Ihr Zielwert bleibt bis zur fachlichen Festlegung `null`.
- Die KI-Ausgabe ist mehrwertig und legt kein Primärversagen fest.
- Das optionale Feld **„Primärversagen manuell auswählen“** speichert Auswahl,
  bewertende Person und Zeitpunkt getrennt von der KI-Originalausgabe.

## Sicherheit

- Referenzinventare werden erst nach Vorschau und ausdrücklicher Bestätigung
  atomar aktualisiert.
- Finder-Schaltflächen öffnen ausschließlich vorhandene, freigegebene Ordner.
- Keine automatische Migration, Annotation, Bildänderung oder Modellabfrage.

## Vereinfachter Evaluationsablauf

- Die dauerhafte Hauptansicht zeigt nur Testreihe, Evaluationsvariante,
  Prüfkörper, KI-Modell, eine kurze Variantenbeschreibung, Bereitschaft und
  den zentralen Button **„Analyse starten“**.
- Technische Bestandszahlen, Kontrollübersicht und die manuellen Diagnoseknöpfe
  sind unter **„Technische Details anzeigen“** eingeklappt.
- Beim zentralen Start erzeugt die App automatisch einen aktuellen Kontrollplan
  und validiert Ordnerstruktur, Variante, Referenzbilder, Beschreibungen sowie
  die Trennung lokaler und cloudbasierter Modelle.
- Nicht bereite Varianten nennen die fehlenden Merkmale und verlinken direkt
  zur Referenzdatenbank; vor erfolgreicher Prüfung wird kein Modell gestartet.
- Technische Plan-, Hash-, Modell- und Varianteninformationen bleiben erhalten
  und werden weiterhin für wissenschaftliche Läufe protokolliert und exportiert.

## Noch offen

- Manuelle Auswahl, Benennung und Kopie der Referenzbilder einschließlich KB für Varianten 2/2a.
- Manuelle Annotation und Beschreibung der Kopien für Variante 3.
- Fachliche Prüfung der Bu-Definition und anschließend gegebenenfalls Aktivierung
  von `require_review` im Manifest von Variante 3.

## Technische Validierung vom 2026-11-02

- Python-Syntaxprüfung: erfolgreich.
- Lokale automatisierte Tests: 112 von 112 erfolgreich.
- Frontend-Produktionsbuild: erfolgreich.
- macOS-Build: erfolgreich; App unter `dist/KI-Bildanalyse.app` erzeugt und
  ad-hoc signiert/verifiziert.
- Installation: `/Applications/KI-Bildanalyse.app`, Signaturprüfung erfolgreich.
- Echter Starttest: installierte App gestartet; Prozess und Startansicht geprüft.
- Sichtprüfung: Beispielprojekt sowie die Register „Referenzdatenbank“ und
  „Evaluation“ in der installierten App geöffnet. Elf Referenzmerkmale,
  Varianten 2/2a/3, die referenzfreie Variante 1, Leerzustände sowie Bu/AA/SSA
  werden angezeigt. Die Evaluation zeigt den zentralen Start, den eingeklappten
  Technikbereich und für Variante 2 die konkrete Liste aller fehlenden Merkmale;
  der Startknopf bleibt dort gesperrt.
- Sicherung der zuvor installierten App:
  `backups/installed-app-before-evaluation-simplification-20261102-135000/KI-Bildanalyse.app`.
- Es wurde kein Modell-, Ollama- oder Cloud-API-Aufruf ausgeführt.

Die vorstehende technische Validierung dokumentiert den Stand vom 02.11.2026 vor
der Erweiterung auf zwölf Merkmale. KB und Regelversion 1.3 werden im aktuellen
Prüflauf vom 03.11.2026 gesondert validiert.

## Regelwerk v1.4 und Evidenzstatus – Prüfung vom 2026-11-04

- Regelwerk v1.3 bytegenau unter
  `Datenbank/Archiv/klassifikationsregeln_v1.3.json` archiviert; SHA-256:
  `7b7f8ad3fe390e17dc14768215dc2ea893e56af180d538477268e2c0f68ea2a2`.
- Regelwerk v1.4 und Promptversion `5.3-qwen-joint-images-evidence-status`
  eingeführt; neue Ergebnisse verwenden Schema 4.1.
- Jeder Vorschlag besitzt zusätzlich zur unveränderten Konfidenz den
  Evidenzstatus `sicher_sichtbar` oder `vermutet`.
- Die Hauptauswertung berücksichtigt beide Evidenzstatus. Die getrennte
  Zusatzsicht für sicher sichtbare Vorschläge weist Abdeckung, ausgeschlossene
  Fälle, Teilmengengenauigkeit und effektive Trefferquote über alle Fälle aus.
- Fachlicher Prüfstatus: fünf Regeln `bedingt_geeignet`, sieben Regeln
  `fachlich_offen`; keine Regel wurde auf bestätigt gesetzt.
- Python-Syntaxprüfung: erfolgreich.
- Automatisierte Tests: 119 von 119 erfolgreich.
- Frontend-Produktionsbuild: erfolgreich.
- Keine KI-Evaluation und kein Modell- oder API-Aufruf ausgeführt.

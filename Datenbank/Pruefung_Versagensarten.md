# Prüfung der Versagensarten

Stand: 21. Oktober 2026  
Geltungsbereich: Zugversuche an Holz parallel zur Faserrichtung

## Ergebnis der JSON-Prüfung

- `versagensarten.json` ist valides JSON und enthält 29 eindeutige Kürzel mit 29 eindeutigen Klassennamen.
- `versagensarten_erweitert.json` ist valides JSON und enthält dieselben 29 Kürzel und Namen, ohne fehlende, zusätzliche oder doppelte Klassen.
- Jeder Eintrag in `failure_modes` besitzt vollständig die Felder `code`, `name`, `feature_type`, `review_required`, `definition`, `visual_features`, `influence_on_failure`, `distinguishing_features`, `limitations` und `sources`. Die Datentypen und Listenbelegungen sind konsistent.
- Technische Einschränkung: Es existiert kein maschinenlesbares JSON-Schema. Quellen stehen nur als Freitext in Listen; DOI/URL, Quellenart, geprüfter Abschnitt und Zugriffsstatus sind nicht strukturiert erfasst.
- 19 der 29 Einträge sind bereits mit `review_required=true` markiert. Die ursprüngliche Datei `versagensarten.json` wurde nicht verändert.

## Geprüfte Klassen

| Kürzel | Klasse | Einordnung | Ergebnis |
|---|---|---|---|
| GSF | Globale Schrägfaser | sichtbares Holzmerkmal | Fachlich plausibel und bedingt bildgeeignet; fotografische Mindestlänge und Winkelkonstanz fehlen. |
| LSF | Lokale Schrägfaser | sichtbares Holzmerkmal | Fachlich plausibel und bedingt bildgeeignet; räumliche Zonengrenze fehlt. |
| KA | Kantenast | sichtbares Holzmerkmal | Geometrische Grunddefinition vorläufig bestätigt; Abgrenzung zu SSA muss laborintern festgelegt werden. |
| SA | Seitenast | sichtbares Holzmerkmal | Grundgeometrie plausibel; die Regel „vollständig innerhalb der Breitseite“ ist als Laborregel zu bestätigen. |
| sp HB | Spröder Holzbruch | sichtbares Bruchbild | Nur als Bruchmorphologie nutzbar; Klassenname und Grenze zu Spl HB sind fachlich offen. |
| Spl HB | Splitterholzbruch | sichtbares Bruchbild | Nur als Bruchmorphologie nutzbar; quantitative Schwelle und Mischbruchregel sind fachlich offen. |

## Fachliche Kontrolle und notwendige Präzisierungen

### GSF und LSF

Das USDA Wood Handbook definiert Faserneigung relativ zur Bauteilachse, nennt Risse, Poren, Holzstrahlen und Harzkanäle als sichtbare Hinweise und beschreibt ausdrücklich, dass Faserabweichung durch Äste lokal auftreten kann. Für die räumliche Gesamtneigung sind benachbarte Flächen zusammen zu beurteilen. Die Studie von Hunger und van de Kuilen bestätigt den Einfluss lokaler Faserabweichung bei Wipfelbrüchen auf Zugfestigkeit, schließt die lokale Astumgebung aber ausdrücklich aus ihrer Untersuchung aus.

Zu korrigieren beziehungsweise einzuschränken:

- Jahrringlinien dürfen nicht allein als Faserrichtung verwendet werden.
- Ein schräger Bruch ist nur ein Indiz, kein Beweis für GSF.
- „Global“ und „lokal“ benötigen eine reproduzierbare Bezugs- beziehungsweise Mindestlänge.
- Die Ursache der Faserabweichung, insbesondere Drehwuchs oder Wipfelbruch, ist aus normalen Prüffotos meist nicht nachweisbar.

### KA und SA

Die eingesehenen Erläuterungen zur DIN 4074-1 unterscheiden den klassischen Seitenast und den an der Kante auftretenden Kantenast und verlangen eine Beurteilung der sichtbaren Flächen. Das stützt die Grundgeometrie. Der verbindliche Normvolltext wurde jedoch nicht eingesehen.

Zu korrigieren beziehungsweise einzuschränken:

- KA benötigt mindestens die Breitseite, die angrenzende Schmalseite und möglichst eine Kantenansicht.
- SA kann erst sicher vergeben werden, wenn ein Übergreifen auf die Kante beziehungsweise Schmalseite ausgeschlossen ist.
- Die Abgrenzung KA/SSA sowie minimale Anschnitte auf der zweiten Fläche müssen im Laborcodebuch festgelegt werden.
- Ein Ast im Bruchbereich ist ein sichtbares Holzmerkmal; seine tatsächliche Ursächlichkeit ist damit nicht bewiesen.

### sp HB und Spl HB

Die Literatur beschreibt bei Zug parallel zur Faser splitternde sowie quer beziehungsweise vergleichsweise glatt erscheinende Brüche. Die im erweiterten JSON genannten makroskopischen Grenzformulierungen wie „deutlich kleiner als die Probenbreite“ oder „mehrfache Probenbreite“ sind durch die eingesehenen Quellen nicht als allgemeingültige Schwellen bestätigt. Die aktuelle ASTM-D143-Produktseite wurde geprüft, der Normvolltext jedoch nicht eingesehen. Deshalb wird keine normative Gleichsetzung behauptet.

Zu korrigieren beziehungsweise einzuschränken:

- Die Klassen dürfen zunächst nur das sichtbare Bruchbild bezeichnen.
- „Spröde“ im mechanischen Sinn erfordert Kraft-Verformungs- beziehungsweise Energiedaten und ist aus Fotos allein nicht nachweisbar.
- Ein splitterndes Bruchbild beweist weder hohe Zähigkeit noch fehlerfreies, gesundes oder hochfestes Holz.
- Für beide Klassen werden Referenzbilder, Messschwellen und eine Regel für gemischte Brüche benötigt.

## Für Bilderkennung geeignete Regeln

- GSF/LSF: Erkennung und räumliche Ausdehnung belastbarer Faserindikatoren; mehrere Längsflächen erforderlich.
- KA/SA: Lage des Astanschnitts relativ zu Längskanten und angrenzenden Flächen; Mehransichten zwingend.
- sp HB/Spl HB: Bruchlänge, Orientierung, Faserauszug, Splitterlänge und Verzahnung; beide Bruchhälften und Maßstab erforderlich.
- Für alle Klassen: `UNKLAR`, wenn benötigte Ansichten fehlen. Holzmerkmal, Bruchbild und Ursache sind als getrennte Annotationsebenen zu speichern.

Nicht als automatische Regel geeignet sind derzeit die Feststellung einer tatsächlichen Bruchursache, die Schätzung der Festigkeitsminderung, die Ableitung von Zähigkeit oder Materialgesundheit und die Unterscheidung innerer Merkmale ohne geeignete Ansichten.

## Offene fachliche Fragen

1. Welche Labordefinition und welche Referenzbilder gelten verbindlich für KA, SA und SSA?
2. Genügt bei KA jeder sichtbare Anschnitt auf der zweiten Fläche, oder gilt eine Mindestgröße?
3. Welche Bezugs- und Mindestlänge trennt GSF von LSF, und welcher Mindestwinkel ist relevant?
4. Welche Oberflächenmerkmale dürfen im vorhandenen Bildbestand als belastbare Faserindikatoren verwendet werden?
5. Soll `sp HB` trotz fehlender Kraft-Verformungs-Daten „Spröder Holzbruch“ heißen oder neutral als kurze/quer verlaufende Bruchmorphologie geführt werden?
6. Welche Messschwellen trennen `sp HB` und `Spl HB` für Holzart, Feuchte, Querschnitt und Versuchsanordnung dieser Arbeit?
7. Wie werden gemischte Brüche annotiert: beide Labels, segmentweise Bruchflächenannotation oder eigene Unsicherheitsklasse?
8. Welche Instanz führt die fachliche Freigabe durch, und wie wird die Interrater-Übereinstimmung eines Referenzbildsatzes geprüft?

## Tatsächlich eingesehene Quellen

- [USDA Forest Products Laboratory: Wood Handbook, Kapitel 5 – Mechanical Properties of Wood (2021)](https://research.fs.usda.gov/download/treesearch/62244.pdf): relevante Abschnitte zu Ästen, Faserneigung und Druckschäden eingesehen.
- [Hunger & van de Kuilen (2018): Slope of grain measurement](https://doi.org/10.1007/s00226-018-1000-7): bibliografische Angaben und Abstract eingesehen; Volltext nicht eingesehen.
- [HFM: Erläuterungen zur DIN 4074-1](https://www.bischoff-schaefer.de/fileadmin/editor/PDF/DE/Datenblaetter/erlaeuterungen_zur_din_4074_1.pdf): Abschnitte zu Asttypen und Faserneigung eingesehen; kein Ersatz für den Normvolltext.
- [Cramer & McDonald (1989): Predicting lumber tensile stiffness and strength with local grain angle measurements](https://wfs.swst.org/index.php/wfs/article/download/1398/1398): Volltext eingesehen.
- [ASTM D143, offizielle Produktseite](https://store.astm.org/d0143-25.html): Anwendungsbereich und aktuelle Ausgabe geprüft; Normvolltext nicht eingesehen.

Weitere in `versagensarten_erweitert.json` genannte Normen, Bücher und Aufsätze wurden in dieser Prüfung nicht vollständig eingesehen. Ihre Nennung wurde daher nicht als inhaltliche Bestätigung gewertet.

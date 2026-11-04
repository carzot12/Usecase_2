# Wissenschaftliche Evaluationsvarianten

Dieser Bereich ist von der internen Evaluation A–D getrennt. Die Ordner enthalten
ausschließlich die für einen wissenschaftlichen Variantenlauf freigegebenen
Referenzbilder. Originalbilder werden weder verschoben noch verändert.

- **Variante 1:** referenzfrei. Es werden weder Referenzordner noch Ground Truth,
  Prüfprotokolle, frühere KI-Ergebnisse oder Expertenbewertungen gelesen.
- **Variante 2:** kleiner, fester Referenzpool (im Manifest: mindestens 1 und
  höchstens 4 Bilder pro Klasse).
- **Variante 2a:** größerer, fester Referenzpool; Zielwerte werden ausschließlich
  im Manifest geändert.
- **Variante 3:** eigener Pool manuell annotierter Bildkopien. Jede Kopie ist
  über `source_reference_id` mit einem unveränderten Original aus Variante 2a
  verknüpft und besitzt eine konkrete bildbezogene Beschreibung.

Referenzbilder werden manuell in die zwölf Merkmalsordner kopiert. Die stabile ID
lautet `<Klassencode>:<relativer Pfad>`, beispielsweise
`KA:KA/beispiel.jpg`. SHA-256-Hashes werden bei der Strukturprüfung neu ermittelt.

Empfohlenes Benennungsschema für unveränderte Referenzen:
`REF_<Merkmal>_<Versuchsserie>_<Prüfkörper>_B<Bildnummer>.jpg`, zum Beispiel
`REF_KA_Z30_031_B01.jpg`. Für manuell annotierte Kopien wird PNG und das Schema
`ANN_<Merkmal>_<Versuchsserie>_<Prüfkörper>_B<Bildnummer>.png` empfohlen.
Andere unterstützte, eindeutige Dateinamen bleiben zulässig.

Bu, AA, SSA und KB sind gleichwertige aktive Merkmale und stehen
in `active_class_codes`. Für sie ist der Zielwert derzeit `null`, also fachlich
noch nicht festgelegt. Das bedeutet nicht null Bilder: Das Minimum bleibt eins.
Alle zwölf Merkmale dürfen gemeinsam als strukturierte Mehrfachmerkmale auftreten. KB bleibt dabei eine neutral behandelte Prüfspur.

Die merkmalsspezifischen Werte stehen in
`reference_requirements_by_feature`. Die bisherigen acht Merkmale behalten in
Variante 2 das Ziel 2 bei maximal 4 Bildern und in Variante 2a/3 das Ziel 20.
Für Bu, AA, SSA und KB wird kein Zielwert erfunden. Der neue KB-Ordner bleibt leer, bis seine Referenzbilder manuell ausgewählt werden; Bilder werden weder kopiert noch automatisch zugeordnet.

Die allgemeinen Klassenbeschreibungen stammen in allen vier Varianten aus
`klassifikationsregeln.json`. Variante 3 dupliziert sie nicht, sondern ergänzt
ausschließlich manuelle Bildannotation und Bildbeschreibung. Sie ist weder
Training noch Fine-Tuning.

`descriptions.csv` verwendet UTF-8 und die Spalten
`reference_id,source_reference_id,class_code,annotated_path,description,reviewed_by,reviewed_at`.
Variante 3 ist erst bereit, wenn jede annotierte Datei genau eine gültige Zeile,
eine nichtleere Beschreibung und eine existierende Quelle aus Variante 2a
besitzt. Wenn `require_review` aktiv ist, sind auch prüfende Person und Datum
Pflicht. Texte und Markierungen werden niemals automatisch erzeugt.

Leere oder unvollständige Ordner verhindern nicht den App-Start; sie markieren
die betreffende Variante lediglich als **NICHT BEREIT**. Nicht unterstützte
Dateien und doppelte Bild-Hashes werden in der Kontrollübersicht ausgewiesen.


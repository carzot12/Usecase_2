# Variante 2A – One-Shot (ein Bild je Merkmal)

In jeden der zwölf Merkmalsordner kommt genau ein manuell ausgewähltes und
fachlich bestätigtes Referenzbild. Das Bild muss auch in Variante 2B enthalten
sein.

Namensbeispiel: `REF_KA_SerieX_017_B01.jpg` (fiktiv, nicht aus der Testserie).
Die Herkunft und fachliche Bestätigung werden in `referenzen.csv` dokumentiert.
Ohne `geprueft_von` und `geprueft_am` bleibt das Bild unbestätigt und Variante
2A ist nicht startbereit.

Bei jeder Analyse werden alle zwölf bestätigten Referenzbilder direkt zusammen
mit den ausgewählten Bildern des unbekannten Prüfkörpers an das gewählte Modell
übertragen. Variante 2A erzeugt und verwendet kein Referenzgedächtnis.

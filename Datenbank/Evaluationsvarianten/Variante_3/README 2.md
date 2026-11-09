# Variante 3 – annotierte Referenzbilder

Variante 3 verwendet je Merkmal genau eine manuell annotierte Kopie eines
fachlich bestätigten Bildes aus Variante 2B. Die Originale aus Variante 2B
bleiben unverändert. Nur Einträge aus `descriptions.csv` werden direkt an das
Modell übertragen; ein Referenzgedächtnis wird nicht verwendet.

Jede Annotation benötigt genau eine Zeile in `descriptions.csv`. Die Spalten
bleiben:

`reference_id,source_reference_id,class_code,annotated_path,description,reviewed_by,reviewed_at`

Namensbeispiel: `ANN_KA_SerieX_017_B01.png` (fiktiv, nicht aus der Testserie).
Die verbindliche Markierungslegende steht im Varianten-Manifest und in
`README.md`.
Die vollständige Ausfüllhilfe steht eine Ebene höher in
`ANLEITUNG_REFERENZBILDER.md`.

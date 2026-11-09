# Manuell annotierte Referenzen für Variante 3

Variante 3 besitzt einen eigenen Bestand manuell erstellter Bildkopien. Die
Originale aus Variante 2B bleiben unverändert. Die App erzeugt, zeichnet oder
verändert keine Annotation und erfindet keine Beschreibung.

Annotierte Kopien werden in den zwölf Merkmalsordnern abgelegt. Jede als
Modellreferenz verwendete Datei benötigt genau eine Zeile in `descriptions.csv`:

`reference_id,source_reference_id,class_code,annotated_path,description,reviewed_by,reviewed_at`

`source_reference_id` verweist auf das unveränderte Ursprungsbild in Variante
2B. `annotated_path` ist relativ zu diesem Ordner, zum Beispiel
`GSF/ANN_GSF_ZA_001_B01.png`. Nur vollständig erfasste und fachlich bestätigte
CSV-Einträge werden direkt an das Modell übertragen. Weitere Bilddateien im
Ordner gelten nicht automatisch als Referenz.

## Markierungslegende

| Markierung | Merkmale |
| --- | --- |
| Roter Kreis | KA, SA, SSA, AA |
| Blaues Rechteck | GSF, LSF, LR |
| Rotes Rechteck | sp HB, Spl HB |
| Lila Rechteck | Bu |
| Grünes Rechteck | RiSch, KB |

Fehlende Dateien, Texte, Quellenverknüpfungen oder Bestätigungen markieren
ausschließlich Variante 3 als **NICHT BEREIT** und verhindern nicht den Start
der Anwendung.

`class_code` akzeptiert GSF, LSF, KA, SA, sp HB, Spl HB, RiSch, LR, Bu, AA, SSA
und KB. Fiktives Beispiel nur für die Dokumentation:

```csv
V3-Bu-001,Bu:Bu/REF_Bu_ZA_019_B01.jpg,Bu,Bu/ANN_Bu_ZA_019_B01.png,"Lila Rechteck markiert das Merkmal Bu.",Fiktive Prüfperson,2026-01-01
```


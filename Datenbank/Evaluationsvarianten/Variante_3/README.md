# Manuell annotierte Referenzen für Variante 3

Variante 3 besitzt einen eigenen Bestand **manuell erstellter Bildkopien**. Die
Originale aus Variante 2a bleiben unverändert. Die App erzeugt, zeichnet oder
verändert keine Annotation und erfindet keine Beschreibung.

Annotierte Kopien werden in den zwölf Merkmalsordnern abgelegt. Jede Datei benötigt
genau eine Zeile in `descriptions.csv` mit:

`reference_id,source_reference_id,class_code,annotated_path,description,reviewed_by,reviewed_at`

`source_reference_id` verweist auf das unveränderte Ursprungsbild in Variante
2a. `annotated_path` ist relativ zu diesem Ordner, zum Beispiel
`GSF/V3-GSF-001.png`. Die CSV-Reihenfolge ist die Modellinput-Reihenfolge.

Fehlende Dateien, Texte, Quellenverknüpfungen oder nicht zugeordnete Bilder
markieren ausschließlich Variante 3 als **NICHT BEREIT** und verhindern nicht
den Start der Anwendung. `require_review` im Manifest steuert, ob zusätzlich
`reviewed_by` und `reviewed_at` ausgefüllt sein müssen.

`class_code` akzeptiert GSF, LSF, KA, SA, sp HB, Spl HB, RiSch, LR, Bu, AA,
SSA und KB. Beispiele nur für die Dokumentation:

```csv
V3-Bu-001,Bu:Bu/REF_Bu_Z30_019_B01.jpg,Bu,Bu/ANN_Bu_Z30_019_B01.png,"Die markierte Zone zeigt das sichtbare Merkmal Buchs.",Carina Zott,2026-11-02
V3-AA-001,AA:AA/REF_AA_Z30_020_B01.jpg,AA,AA/ANN_AA_Z30_020_B01.png,"Markiert sind drei räumlich eng beieinanderliegende Äste.",Carina Zott,2026-11-02
V3-SSA-001,SSA:SSA/REF_SSA_Z30_021_B01.jpg,SSA,SSA/ANN_SSA_Z30_021_B01.png,"Der vollständig auf der Schmalseite sichtbare Ast ist markiert.",Carina Zott,2026-11-02
```


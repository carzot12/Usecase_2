# Anleitung für Referenzbilder

## Bilder ablegen

Lege jedes Bild in den Ordner des Merkmals, das es zeigt. Für `sp HB` heißt der
Ordner `sp_HB`, für `Spl HB` heißt er `Spl_HB`. Alle anderen Ordner tragen den
Merkmalscode.

- Variante 2A enthält genau ein Bild je Merkmal: insgesamt 12 Bilder.
- Variante 2B enthält als Ziel drei Bilder je Merkmal: insgesamt 36 Bilder.
- Das Bild aus 2A muss auch im passenden Ordner von 2B enthalten sein.
- Zu jedem Bild aus 2B braucht Variante 3 genau eine annotierte Kopie: bei
  Zielbestand insgesamt 36 Bilder.

Verwende für unveränderte Referenzen das Schema
`REF_<Merkmal>_<Serie>_<Prüfkörper>_B<Nr>.jpg`, für annotierte Kopien
`ANN_<Merkmal>_<Serie>_<Prüfkörper>_B<Nr>.png`.

Die 49 Prüfkörper der Testserie Z-30 sind als Referenzmaterial ausgeschlossen.

## Herkunft dokumentieren

Trage Bilder von 2A und 2B in der jeweiligen `referenzen.csv` ein. Ersetze dabei
alle Platzhalter durch die wirklichen Angaben; leere oder erfundene Angaben sind
nicht zulässig.

Variante 3 verwendet `descriptions.csv`. `source_reference_id` ist die stabile ID
des unveränderten 2B-Bildes: `<Merkmal>:<Merkmalsordner>/<Dateiname>`.
`annotated_path` ist der Pfad der PNG-Kopie innerhalb von Variante 3.

Fiktives Beispiel (keine echte Referenz und nicht aus der Testserie):

```csv
ANN-KA-001,KA:KA/REF_KA_SerieX_017_B01.jpg,KA,KA/ANN_KA_SerieX_017_B01.png,"Die Markierung kennzeichnet den sichtbaren Kantenast.",Beispielperson,2099-01-01
```

Jedes 2B-Bild darf genau einmal als Quelle verwendet werden. Dateiname, Merkmal
und Quelle müssen zusammenpassen.

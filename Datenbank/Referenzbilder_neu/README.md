# Referenzbilder für die aktive Zugversuchs-Klassifikation

Dieser bestehende Ordner ist der beschreibbare Referenzbildspeicher der Desktop-App.

- `aktive_hauptversagensarten.json` dokumentiert die aktuell acht in `Datenbank/klassifikationsregeln.json` beschriebenen Hauptklassen: GSF, LSF, KA, SA, sp HB, Spl HB, RiSch und LR. Die Anwendung leitet diese Auswahl aus den vorhandenen Regeln ab; spätere fachlich dokumentierte Regeln können ergänzt werden.
- `referenzbilder_manifest.json` wird erst beim ersten Import oder bei der ersten fachlichen Bearbeitung angelegt. Es dokumentiert Klasse, Prüfkörper-ID, Anmerkung, Bestätigungsstatus, SHA-256, Quellpfad und gespeicherten Pfad.
- Bereits vorhandene Bilder werden zunächst nur als unregistriert und unbestätigt angezeigt. Die App verändert oder löscht sie nicht.
- Hashgleiche Bilder werden beim Import nicht erneut kopiert.
- Bestätigte Einträge können nur nach einer zusätzlichen ausdrücklichen Bestätigung geändert werden. Es gibt keine Löschfunktion.

Die wissenschaftliche Evaluationsvariante 1 greift weiterhin nicht auf Referenzbilder zu. Dieser Verwaltungsspeicher wird nicht automatisch als Ground Truth oder als Modellinput verwendet.

# Wissenschaftliche Evaluationsvarianten

Die Varianten sind vom Testdatensatz und von der Ground Truth getrennt:

- **Variante 1:** Zero-Shot ohne Referenzbilder.
- **Variante 2A:** One-Shot mit genau einem bestätigten Referenzbild je Merkmal.
- **Variante 2B:** Few-Shot mit dem fachlich bestätigten festen Referenzpool.
- **Variante 3:** annotierte Kopien desselben 2B-Pools; alle bestätigten Bilder
  werden direkt verwendet. Nur `Variante_3` ist aktiv, `Variante_3_alt` ist Archiv.

Alle Varianten verwenden dieselben zwölf Merkmale. Leere oder unvollständige
Ordner verhindern den App-Start nicht. Nur die jeweilige Variante wird als
**NICHT BEREIT** angezeigt.

Die Referenzbilder werden ausschließlich manuell eingefügt. Einzelheiten stehen
in [ANLEITUNG_REFERENZBILDER.md](ANLEITUNG_REFERENZBILDER.md).

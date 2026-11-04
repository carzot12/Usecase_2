# Entwicklerdokumentation – KI-Bildanalyse (Use Case 2)

**Stand:** 02.11.2026 · App-Version 1.0.0 · Regelversion 1.2 · Promptversion `5.2-qwen-joint-images-primary-candidate` · Ergebnisschema 4.0
**Zielgruppe:** Personen, die die Anwendung weiterentwickeln, bauen, testen oder übernehmen.

Architekturdiagramme: [Systemarchitektur_KI_Bildanalyse.md](Systemarchitektur_KI_Bildanalyse.md). Fachliche Einordnung: [Technischer_Bericht_KI_Bildanalyse.md](Technischer_Bericht_KI_Bildanalyse.md).

> **Wichtig vor jeder Änderung:** Der aktuelle Funktionsstand (Regeln 1.2, Variante 1, Referenzspeicher) ist **nicht committet** (letzter Commit `0504645`). Bitte zuerst committen, damit Prompt-Hashes späterer Läufe einem Codestand zugeordnet werden können.

---

## 1. Schnellstart

```bash
cd ~/Desktop/Masterarbeit/Usecase_2

# Python-Abhängigkeiten (openai>=2.0.0, Pillow>=10.0.0, pywebview>=6.1)
python3 -m pip install -r requirements-app.txt

# Frontend
npm --prefix frontend ci            # reproduzierbar über package-lock.json
npm --prefix frontend run build     # tsc -b && vite build → frontend/dist/

# App starten
python3 app.py

# Frontend mit Hot Reload
npm --prefix frontend run dev                     # Vite auf 127.0.0.1:5173
python3 app.py --dev-url http://127.0.0.1:5173    # aktiviert pywebview-Debug

# Tests (keine Netzwerk- oder Modellaufrufe)
python3 -m unittest discover -s tests -v
```

Getestete Umgebung beim letzten Prüflauf: macOS (arm64), Python 3.12.9, Node.js 26.7.0, Pillow 12.3.0, openai 2.53.0; Frontend: React 19.3.1, Vite 8.3.5, TypeScript 7.0.2.

Für lokale Analysen muss Ollama laufen (`ollama serve`) und
`qwen3-vl:4b-instruct` verfügbar sein. Die App startet Ollama nicht. Die elf
aktiven Merkmalscodes werden zentral aus
`Datenbank/klassifikationsregeln.json` (`meta.active_class_codes`) geladen.
Prompt 5.2 wertet alle ausgewählten Ansichten gemeinsam aus, fordert genau
einen unverbindlichen visuellen Primärkandidaten bei `KLASSIFIZIERT` und trennt
`UNKLAR`, `KEINE_MERKMALE` und technische Validierungsfehler.

---

## 2. Modulübersicht

| Modul | Verantwortung | Wichtige Symbole |
|---|---|---|
| [app.py](../app.py) | Einstiegspunkt | `main()` |
| [desktop/main.py](../desktop/main.py) | pywebview-Fenster, Splash, Ressourcenpfade | `resource_path`, `frontend_url`, `launch`, `SPLASH_MINIMUM_SECONDS = 7.0` |
| [desktop/bridge.py](../desktop/bridge.py) | JS-API, Projekt-/Auftrags-/Bewertungs-/Exportlogik | `DesktopBridge`, `MODEL_CONFIGS`, `WORKSPACE_ROOT`, `APP_DATA`, `DEFAULT_SELECTED_IMAGE_COUNT = 3` |
| [desktop/reference_store.py](../desktop/reference_store.py) | App-Referenzspeicher `Datenbank/Referenzbilder_neu/` | `ReferenceStore` |
| [Tool/ki_bildanalyse.py](../Tool/ki_bildanalyse.py) | Klassifikationskern und CLI | `load_catalog`, `collect_test_images`, `select_references`, `_analysis_instructions`, `_prompt_catalog`, `build_request_content`, `build_ollama_messages`, `build_mistral_messages`, `response_schema`, `call_ollama`, `call_openai_with_metadata`, `call_mistral`, `validate_and_enrich`, `build_result_document`, `save_result`, `run_analysis` |
| [Tool/openai_config.py](../Tool/openai_config.py) | OpenAI-Schlüssel und Fehler | `resolve_api_key`, `create_openai_client`, `safe_request_error`, `KEYCHAIN_SERVICE = "KI-Bemessungstool OpenAI"` |
| [Tool/mistral_config.py](../Tool/mistral_config.py) | Mistral-Schlüssel und geschwärzte Fehler | `resolve_mistral_api_key`, `safe_mistral_request_error`, `KEYCHAIN_SERVICE = "Usecase_2_Mistral"` |
| [Tool/evaluation.py](../Tool/evaluation.py) | Evaluation A–D (CLI) | `build_ground_truth_document`, `load_ground_truth`, `select_holdout_references`, `build_variant_content`, `execute_job`, `compute_metrics`, `run_evaluation` |
| [Tool/evaluation_variants.py](../Tool/evaluation_variants.py) | Wissenschaftliche Varianten 1, 2, 2a und 3, Strukturprüfung, atomare Inventare und Referenzgedächtnis | `inspect_variant`, `reference_inventory_preview`, `update_reference_inventories`, `deterministic_batches`, `prepare_reference_memory` |
| [Tool/bildzuordnung_neue_analysen.py](../Tool/bildzuordnung_neue_analysen.py) | Sortierung neuer Serien (CLI) | `load_photos`, `recognize_markers`, `build_assignments`, `write_outputs` |
| [frontend/src/App.tsx](../frontend/src/App.tsx) | gesamte Oberfläche | `App`, `RunCard`, `modelChoices`, `tabs`, `api()` |

---

## Wissenschaftliche Varianten

- Variante 1 liest keine Referenz- oder Annotationsordner und bleibt auch bei
  fehlenden Ordnern funktionsfähig.
- Variante 2 und 2a lesen ausschließlich ihre eigenen, unveränderten Bildpools.
  Zielmengen kommen aus dem jeweiligen `variant_manifest.json`.
- Variante 3 liest ausschließlich manuell annotierte Kopien aus ihren eigenen
  Klassenordnern. `descriptions.csv` verknüpft jede Annotation über
  `source_reference_id` mit einem Original aus 2a. Allgemeine
  Klassenbeschreibungen bleiben getrennt und kommen aus
  `klassifikationsregeln.json`.

Die CSV-Spalten sind
`reference_id,source_reference_id,class_code,annotated_path,description,reviewed_by,reviewed_at`.
Die CSV-Reihenfolge ist die Reihenfolge der Referenzinputs. Validiert werden IDs,
Klasse, Pfad, Dateiexistenz, Beschreibung, Quellenlink, getrennte Dateien,
SHA-256, Duplikate und – abhängig von `require_review` – Prüfperson und Datum.
Annotationen werden nicht erzeugt; der Ablauf ist Referenzvorbereitung, kein
Training und kein Fine-Tuning.

---

## 3. Konfiguration

### 3.1 Konstanten in `Tool/ki_bildanalyse.py`

| Konstante | Wert | Bedeutung |
|---|---|---|
| `SUPPORTED_SUFFIXES` | `.jpg .jpeg .png .webp .gif` | analysierbare Formate |
| `MAX_TRANSMITTED_LONG_EDGE` | 2048 | max. Kante für OpenAI/Mistral |
| `DEFAULT_OLLAMA_MAX_IMAGE_EDGE` | 768 | max. Kante für Ollama |
| `DEFAULT_REFERENCES_PER_CLASS` | 1 | Referenzen je Klasse (Standardmodus) |
| `DEFAULT_OLLAMA_URL` | `http://localhost:11434` | erzwungen |
| `DEFAULT_OLLAMA_TIMEOUT_SECONDS` | 900 | Zeitlimit Ollama |
| `DEFAULT_OLLAMA_MODEL` | `qwen3-vl:4b` | |
| `DEFAULT_OLLAMA_NUM_CTX` / `_NUM_PREDICT` | 24576 / 4096 | |
| `DEFAULT_MISTRAL_MODEL` | `ministral-3b-2512` | |
| `DEFAULT_MISTRAL_URL` | `https://api.mistral.ai/v1/chat/completions` | erzwungen |
| `DEFAULT_MISTRAL_TIMEOUT_SECONDS` | 900 | |
| `MAX_MISTRAL_IMAGES_PER_REQUEST` | 8 | inkl. Referenzen |
| `PROMPT_VERSION` | `3.1-eight-class-visual-only` | **manuell erhöhen**, wenn sich Promptlogik ändert |
| `SYSTEM_INSTRUCTIONS` | Systemprompt | |
| `CAUSAL_CLAIM` | regulärer Ausdruck | Kausalitätsfilter |
| `ALLOWED_INPUT_ROOT` | `Pruefdaten/Neue_Analysen` | Standard-Wurzel für CLI |

### 3.2 Modellauswahl in der App

`MODEL_CONFIGS` in [desktop/bridge.py:44-48](../desktop/bridge.py#L44-L48) und `modelChoices` in [frontend/src/App.tsx:14-18](../frontend/src/App.tsx#L14-L18) müssen synchron gehalten werden. Die Tupelwerte `(num_ctx, num_predict)` werden nur für Ollama ausgewertet.

### 3.3 Umgebungsvariablen

| Variable | Wirkung |
|---|---|
| `EKI_PROJECT_ROOT` | Arbeitsverzeichnis mit `Pruefdaten/` (für Bundle oder abweichenden Ort) |
| `EKI_FRONTEND_URL` | lädt Frontend von URL statt `frontend/dist` |
| `EKI_WEB_DEBUG` | pywebview-Debugmodus |
| `OPENAI_API_KEY` | OpenAI-Schlüssel (Vorrang vor Schlüsselbund) |
| `USECASE2_OPENAI_MODEL` | Standardmodell OpenAI in der CLI (Standard `gpt-6.1-sol`) |
| `USECASE2_OLLAMA_MODEL` | Standardmodell Ollama in der CLI |

Mistral liest den Schlüssel **nur** aus dem Schlüsselbund:

```bash
security add-generic-password -s Usecase_2_Mistral -a "$USER" -w   # interaktive Eingabe
```

### 3.4 Modellverifikationsdateien

`Tool/model_verification.json` und `Tool/mistral_model_verification.json` müssen einen Eintrag mit `"model": <Kennung>` und `responses_api`/`chat_completions_api` sowie `image_input: true` enthalten, sonst verweigert `run_analysis` den Cloud-Aufruf. Bei Modellwechsel zuerst diese Dateien anhand der Anbieterdokumentation aktualisieren.

---

## 4. Bridge-Schnittstelle (JS ↔ Python)

Aufruf im Frontend: `await window.pywebview.api.<name>(payload)`. Rückgabe immer ein Objekt mit `ok`.

| Methode | Payload | Antwort (bei `ok: true`) |
|---|---|---|
| `bootstrap()` | – | `exampleAvailable, examplePath, models[{provider,model,local}], classes[{code,name}], evaluation{activeVariant, ready, reason}` |
| `open_example()` | – | `project` |
| `choose_project()` | – (Ordnerdialog) | `project` oder `cancelled` |
| `create_project({name})` | | `project` |
| `project_snapshot()` | – | `project{name,path,specimens[{id,imageCount,analysisImageCount,identificationImageCount,runCount,hasResult}]}` |
| `specimen_detail({specimen})` | | `images[{name,src,analysisSelected,analysisEligible,isIdentification}], result, resultPath, correction, history[RunSummary]` |
| `create_specimen({specimen})` | | `project, specimen` |
| `import_images({specimen})` | – (Dateidialog) | `imported[]` |
| `save_image_selection({specimen, selectedImages[]})` | | `selectedImages[]` |
| `start_analysis({specimen, provider, model, confirmPaid?, evaluationVariant?, evaluationPlanToken?, allowedClassCodes?})` | | `jobId`; Fehler ggf. mit `needsConfirmation` |
| `analysis_status({jobId})` | | `state (running/complete/error), stage, status, provider, model, specimen, startedAt, finishedAt?, path?, result?, error?` |
| `plan_batch({specimenIds[], models[{provider,model}]})` | | `plan{jobCount, cloudRequestCount, jobs[]}` |
| `start_batch({specimenIds[], models[], confirmCloud})` | | `batchId, batch` |
| `batch_status({batchId})`, `list_batches()` | | `batch` bzw. `batches[]` |
| `cancel_batch_jobs({batchId, jobIds?[]})` | | `batch` |
| `resume_batch({batchId, retryFailed?})` | | `batchId, batch` |
| `dismiss_batch_jobs({batchId, jobIds[]})` | | Blendet ausschließlich terminale Aufträge (`succeeded`, `failed`, `cancelled`) aus; Artefakte und Batch-Plan bleiben erhalten |
| `clear_batch_history()` | | Blendet projektweit alle terminalen Batch-Aufträge aus; laufende, wartende und unterbrochene Aufträge bleiben sichtbar |
| `project_results()` | – | `results[RunSummary]` (neueste zuerst) |
| `result_detail({runId})` | | `runId, path, result, assessment, runImages[]` |
| `save_assessment({runId, verdict, primaryCode, secondaryCodes[], note})` | | `assessment, path` |
| `evaluation_overview()` | – | `activeVariant, classes[], variant1{…}, requestedVariants[], legacyVariants[]` |
| `evaluation_run_plan({specimen})` | | `plan{variant, specimen, images[], imageCount, provider, model, classes[], referenceImages:0, …, planToken}` |
| `reference_catalog()` | – | `configured, activeCodes[], documentedCodes[], classes[{code,name,definition,featureType,references[],confirmedCount,…}], storageRoot, manifestPath` |
| `choose_reference_images({code, specimenId, description, confirmed})` | – (Dateidialog) | `imported[], duplicates[], catalog` |
| `update_reference_entry({id, code, specimenId, description, confirmed, confirmConfirmedChange})` | | `entry, catalog` |
| `export_selected_runs({runIds[]})` | | `jsonPath, csvPath` |

Nicht von der Oberfläche genutzt, aber vorhanden: `save_correction`, `export_overview`, `export_run`, `export_comparison`, `export_specimen`, `import_reference_images`, `save_active_reference_codes`, `open_project_path`.

**Nebenläufigkeit:** Jobs und Batches laufen als Daemon-Threads; gemeinsame Zustände sind über `threading.RLock` geschützt. Batches arbeiten Aufträge **sequenziell** ab. Variante-1-Plan-Tokens liegen nur im Arbeitsspeicher (`self.evaluation_plans`) und verfallen beim App-Neustart.

---

## 5. Kern-API `run_analysis`

```python
from pathlib import Path
from Tool.ki_bildanalyse import run_analysis

destination: Path = run_analysis(
    input_dir,                      # Prüfkörperordner
    provider="ollama",              # "ollama" | "openai" | "mistral"
    model=None,                     # Standardmodell je Provider
    live=False,                     # Pflicht für echte Cloud-Aufrufe
    confirm_paid_request=False,     # zusätzlich Pflicht für Cloud
    max_references_per_class=1,     # in Variante 1 erzwungen 0
    ollama_num_ctx=24576, ollama_num_predict=4096, ollama_max_image_edge=768,
    allowed_input_root=ALLOWED_INPUT_ROOT,
    selected_image_names=None,      # explizite Auswahl (App)
    output_dir=DEFAULT_OUTPUT_DIR,
    progress_callback=None,         # erhält "Modell wird geladen" | "Bilder werden analysiert" | "Ergebnis wird verarbeitet"
    run_id=None, project_id=None,
    evaluation_variant=None,        # None | "1"
    allowed_class_codes=None,       # muss – falls gesetzt – exakt den dokumentierten Codes entsprechen
    client=None, ollama_opener=urlopen, mistral_opener=urlopen,
    mistral_key_resolver=resolve_mistral_api_key,   # für Tests injizierbar
)
```

Ablauf: Variante prüfen → aktive Codes aus Regeln ableiten → Katalog laden (mit/ohne Referenzen) → Bilder prüfen → Referenzen wählen → Prompt bauen → Provider aufrufen → validieren → Hashes und Einstellungen → Ergebnis schreiben.

Für Tests sind Netzwerk und Schlüssel über `client`, `ollama_opener`, `mistral_opener` und `mistral_key_resolver` injizierbar.

---

## 6. Kommandozeilenwerkzeuge

### 6.1 `Tool.ki_bildanalyse`

```bash
# Lokal (Ollama muss laufen) – Standard-Provider ist ollama!
python3 -m Tool.ki_bildanalyse <PK-Ordner> --provider ollama --model qwen3-vl:4b

# Variante 1 (referenzfrei)
python3 -m Tool.ki_bildanalyse <PK-Ordner> --provider ollama --evaluation-variant 1

# Mock ohne Netzwerk (OpenAI/Mistral ohne --live)
python3 -m Tool.ki_bildanalyse <PK-Ordner> --provider mistral

# Cloud – nur nach ausdrücklicher Kostenfreigabe
python3 -m Tool.ki_bildanalyse <PK-Ordner> --provider openai --live --confirm-paid-request
```

Weitere Optionen: `--output-dir`, `--mock-response`, `--max-references-per-class`, `--allowed-class-code` (mehrfach), `--ollama-num-ctx`, `--ollama-num-predict`, `--ollama-max-image-edge`. Der Eingabeordner muss unter `Pruefdaten/Neue_Analysen` liegen. Rückgabewert 0 bei Erfolg, 2 bei Fehler.

### 6.2 `Tool.evaluation`

```bash
python3 -m Tool.evaluation prepare --validate-only                 # Ground Truth prüfen, nichts schreiben
python3 -m Tool.evaluation prepare                                  # schreibt ground_truth_manifest.json (nie überschreiben)
python3 -m Tool.evaluation run --provider ollama --model qwen3-vl:4b               # nur Laufplan
python3 -m Tool.evaluation run --provider ollama --model qwen3-vl:4b --execute     # 90 lokale Aufrufe
python3 -m Tool.evaluation run --provider openai --model gpt-6.1-sol --execute --live --confirm-paid-request
```

### 6.3 `Tool.bildzuordnung_neue_analysen`

```bash
python3 -m Tool.bildzuordnung_neue_analysen <Ordner mit Rohfotos> --dry-run
python3 -m Tool.bildzuordnung_neue_analysen <Ordner mit Rohfotos>
```

Benötigt `Datenbank/Alte_Versuche/Zugversuch 2/bildzuordnung_serien.py` (lokale OCR, EXIF, Farbanalyse). Serienkennung `Z-30` ist fest codiert.

---

## 7. Datenstrukturen

### 7.1 `Datenbank/klassifikationsregeln.json` (v1.2)

```text
meta
  titel, version "1.2", stand, gueltigkeitsbereich
  archiv_vorversion {datei, sha256}
  fotodokumentation[], grundprinzip[], pruefstatus_werte{}, quellenpruefung[]
classes[]                       Reihenfolge = Reihenfolge der aktiven Codes
  code, name, klassentyp, klassifikationsziel
  positive_bildmerkmale[], abgrenzung_zu_aehnlichen_klassen[]
  nicht_vergeben_wenn[], unklar_wenn[], einschraenkungen_visuelle_erkennung[]
  nicht_zulaessige_ursachenaussage
  fachlicher_pruefstatus {status, begruendung, offene_punkte[]}
  literaturquellen[]
```

Jeder `code` muss in `versagensarten_erweitert.json` existieren. Listenfelder müssen nicht leere Textlisten sein.

### 7.2 `Datenbank/versagensarten_erweitert.json` (v1.0)

`failure_modes[]` mit genau 29 Einträgen: `code, name, feature_type (wood_feature|fracture_pattern|failure_mechanism|finger_joint|test_artifact), review_required, definition, visual_features[], influence_on_failure, distinguishing_features[], limitations[], sources[]`.

### 7.3 Kuratiertes Referenzmanifest `Datenbank/Referenzbilder/referenzbilder_manifest.json` (Schema 2.0)

```text
meta {schema_version, aktualisiert_am, klassifikationsregeln_version, context_images_default_for_inference}
dataset_assignments {<PK>: {dataset_role, excluded_from_independent_test, assigned_at}}
reference_sets[]
  id, failure_mode, class_name, specimen, series, dataset_role, excluded_from_independent_test,
  classification_rules_version, confirmation_source
  reference_image {image_role, confirmed, default_for_inference, confirmed_by, confirmed_at,
                   visibility, review_comment, original_path, copied_path}
  context_images[] {image_role, context_kind, confirmed, default_for_inference,
                    assignment_status, original_path, copied_path}
```

Pfadkonvention der Kopien: `Datenbank/Referenzbilder/<Code>_<Name>/<PK>/reference_image/<Datei>`. `select_references` leitet die Prüfkörper-ID aus `path.parents[1]` ab – diese Struktur nicht ändern.

### 7.4 App-Referenzspeicher `Datenbank/Referenzbilder_neu/`

- `aktive_hauptversagensarten.json`: `schemaVersion, requiredCount, selectedCodes[], configured, source, sourceVersion, note` (informativ; maßgeblich sind die Regeln).
- `referenzbilder_manifest.json` (entsteht beim ersten Import/Bearbeiten): `schemaVersion, updatedAt, entries[{id, code, specimenId, description, confirmed, confirmedBy, confirmedAt, originalFilename, sourcePath, storedPath, sha256, registered, createdAt, updatedAt}]`.
- Ablage: `<Code>_<Name>/<PK>/<Datei>`.

### 7.5 Projektdaten `Output/EKI_Bildanalyse/Projekte/<Projekt>/`

| Datei | Struktur |
|---|---|
| `project.json` | `name, source, updatedAt` |
| `Bildauswahl/<PK>.json` | `specimen, selectedImages[], knownImages[], selectionCustomized, updatedAt` |
| `Ergebnisse/<PK>_<provider>_<model>_<YYYYMMDDTHHMMSSZ>_<runId>.json` | Ergebnisschema 2.0 (Technischer Bericht, Anhang B) |
| `Bewertungen/<runId>.json` | `runId, projectId, specimenId, verdict, primaryCode, secondaryCodes[], note, resultPath, updatedAt` |
| `Korrekturen/<PK>.json` | Altformat: `specimen, primaryCode, secondaryCodes[], note, confirmed, createdAt, originalResult` |
| `Batches/<batchId>.json` | Plan + `batchId, createdAt, updatedAt, status, confirmCloud, sequential, jobs[{jobId, runId, specimen, provider, model, selectedImages[], plannedRequestCount, status, stage, statusText, error, startedAt, finishedAt, resultPath, primaryCode}]` |
| `Export/<Projekt>_Bildanalysen_<Zeit>.json/.csv` | siehe Technischer Bericht 11.5 |

Ergebnisdateien werden über das Präfix `<PK>_` einem Prüfkörper zugeordnet (`glob(f"{specimen}_*.json")`). Prüfkörper, deren Namen Präfix eines anderen sind (z. B. `Z-1` und `Z-1_b`), würden sich Ergebnisse teilen – bei der Benennung vermeiden.

### 7.6 Evaluation

`Evaluation/ground_truth_manifest.json`: `schema_version, dataset_name, dataset_kind, created_at, source_manifest, source_manifest_sha256, classification_rules_version, ground_truth_policy, limitations[], case_count, covered_class_count, class_distribution, cases[{case_id, specimen_id, expected_primary_code, expected_primary_name, test_images[{path, sha256}], label_confirmation{source, confirmed_by, confirmed_at, visibility}}]`. Die Datei ist per `.gitignore` von Git ausgeschlossen.

---

## 8. Erweiterungsleitfaden

### 8.1 Neue Klasse aktivieren

1. Code muss bereits in `versagensarten_erweitert.json` existieren (sonst zuerst dort ergänzen; `load_catalog` verlangt exakt 29 Einträge – bei einer 30. Klasse diese Prüfung in [Tool/ki_bildanalyse.py:131](../Tool/ki_bildanalyse.py#L131) anpassen).
2. Vorherige Regeldatei nach `Datenbank/Archiv/` kopieren, SHA-256 in `meta.archiv_vorversion` eintragen, `meta.version` und `stand` erhöhen.
3. Regel mit allen Pflichtfeldern in `classes[]` einfügen (Position = Reihenfolge im Prompt).
4. `PROMPT_VERSION` erhöhen.
5. Fest codierte Texte anpassen: Banner in [App.tsx:102](../frontend/src/App.tsx#L102), Fußzeile [App.tsx:111](../frontend/src/App.tsx#L111), Texte „acht“ in [bridge.py:89](../desktop/bridge.py#L89) und [bridge.py:365](../desktop/bridge.py#L365), `Datenbank/Referenzbilder_neu/aktive_hauptversagensarten.json`, READMEs.
6. Tests anpassen (`test_version_1_2_has_eight_uniform_rules_and_archives_1_1` u. a.), Beispielprompt neu erzeugen.
7. App neu bauen – das Bundle enthält eine Kopie der Regeldateien.

### 8.2 Neues Modell anbinden

1. Eintrag in `MODEL_CONFIGS` und `modelChoices`.
2. Bei Cloud: Eintrag in der Verifikationsdatei.
3. Bei neuem Provider: Builder-Funktion für Nachrichten, `call_<provider>` mit Rückgabe `(raw, metadata)`, Verzweigung in `run_analysis`, Kennzeichen in `build_result_document.execution`, Kostenfreigabe in `start_analysis` und `_planned_batch`, Tests mit Fake-Transport.

### 8.3 Evaluationsvarianten pflegen

Die vier wissenschaftlichen Varianten sind in
`Datenbank/Evaluationsvarianten/Variante_<ID>/variant_manifest.json` definiert.
Mengen werden nur im Manifest geändert. Für Variante 3 müssen annotierte Kopien
manuell in den Klassenordnern abgelegt und vollständig in `descriptions.csv`
verknüpft werden. Vor einem Manifestabgleich immer
`reference_inventory_preview` ausführen; `update_reference_inventories` darf nur
nach ausdrücklicher Bestätigung mit `confirmed=True` aufgerufen werden. Bereits
gespeicherte Laufartefakte und eingefrorene Testreihen werden nicht aktualisiert.

---

## 9. Tests

| Datei | Fälle | Hilfsmittel |
|---|---|---|
| `tests/test_ki_bildanalyse.py` | 33 | temporäre Ordner unter `Pruefdaten/Neue_Analysen`, Fake-`opener`, `SimpleNamespace`-Clients, gepatchte Schlüsselbund-Runner |
| `tests/test_desktop_bridge.py` | 20 | `patch.object(bridge_module, "APP_DATA", …)`, gepatchtes `run_analysis` |
| `tests/test_evaluation.py` | 8 | temporäre Ausgabeordner, gepatchte Provider |
| `tests/test_reference_store.py` | 4 | temporäre Projektwurzel |

Letzter Lauf: 02.11.2026, **103 Tests, alle bestanden** (22,9 s). Die Varianten-3-Tests verwenden ausschließlich temporäre Dateien und Fake-Transports; echte Modell- oder API-Aufrufe werden nicht gestartet.

Nicht abgedeckt: Frontend, Dateidialoge, echte Provider, `bildzuordnung_neue_analysen.py`, Build. Ein Frontend-Testframework ist nicht eingerichtet; Typprüfung: `cd frontend && npx tsc -p tsconfig.app.json --noEmit`.

---

## 10. Build und Auslieferung

```bash
python3 -m pip install -r requirements-build.txt    # zusätzlich pyinstaller==6.16.0
./build_macos.sh                                     # optional PYTHON_BIN=/pfad/zu/python
```

Ablauf des Skripts: Interpreter mit PyInstaller suchen → `npm ci` → Frontend-Build → Pflichtressourcen prüfen → AppleDouble-Dateien und xattr entfernen → `PyInstaller KI-Bildanalyse.spec` → Kopie in `/tmp`, Ad-hoc-Signatur, `codesign --verify --deep --strict` → signiertes Bundle nach `dist/KI-Bildanalyse.app`.

Ins Bundle aufgenommen (`KI-Bildanalyse.spec`): `frontend/dist`, `assets/logo.png`, `assets/widget.png`, `Datenbank/klassifikationsregeln.json`, `Datenbank/versagensarten_erweitert.json`, beide Verifikationsdateien, das kuratierte Referenzmanifest und alle darin referenzierten Hauptreferenzbilder. **Nicht** enthalten: `Tool/mock_response.json` (der Mock-Modus ist in der App nicht erreichbar), Prüfdaten, `Referenzbilder_neu`.

Installation: `dist/KI-Bildanalyse.app` ohne erweiterte Attribute nach `/Applications` kopieren, z. B. `ditto --noextattr --noqtn dist/KI-Bildanalyse.app /Applications/KI-Bildanalyse.app`, anschließend `codesign --verify --deep --strict /Applications/KI-Bildanalyse.app`. (Ein Installationsskript existiert nicht; der Befehl ist ein Vorschlag.) Im synchronisierten Desktop-Ordner schlägt die strenge Prüfung von `dist/` wegen `com.apple.FinderInfo` fehl; die installierte Kopie war beim letzten Prüflauf gültig.

Bekannte Build-Hinweise: Ad-hoc-Signatur ohne Notarisierung; `package.json` mit `latest`; Bundle enthält CPython-3.13-Module (Interpreterwahl im Skript: zuerst `python`, dann `python3`); zahlreiche Bundle-Duplikate in `dist/` können gelöscht werden, nachdem geprüft wurde, dass sie nicht benötigt werden.

---

## 11. Bekannte technische Auffälligkeiten

| Nr. | Befund | Ort |
|---|---|---|
| 1 | Banner-Text nennt „Sieben“ statt acht aktive Klassen | [App.tsx:102](../frontend/src/App.tsx#L102) |
| 2 | Fehlermeldung nennt „eine der 29 Klassen“ | [ki_bildanalyse.py:978](../Tool/ki_bildanalyse.py#L978) |
| 3 | `Tool/README.md` beschreibt Mock als Standard; Standard-Provider der CLI ist `ollama` | [ki_bildanalyse.py:1465](../Tool/ki_bildanalyse.py#L1465) |
| 4 | `Tool/README.md` meldet keinen Mistral-Live-Lauf nach der 422-Korrektur; Ergebnisdateien mit `mode: live` danach existieren | `Output/EKI_Bildanalyse/Projekte/Test_001/Ergebnisse/` |
| 5 | Variante 1 überspringt die Leakage-Prüfung gegen Referenz-/Kontextbilder | [ki_bildanalyse.py:200-208](../Tool/ki_bildanalyse.py#L200-L208) |
| 6 | OpenAI ohne Temperatur und ohne explizites Timeout | [ki_bildanalyse.py:606-619](../Tool/ki_bildanalyse.py#L606-L619) |
| 7 | Bewertungen werden ohne Historie überschrieben; keine Person erfasst | [bridge.py:1023-1037](../desktop/bridge.py#L1023-L1037) |
| 8 | `save_assessment` akzeptiert 29 Codes, Oberfläche bietet 8 | [bridge.py:1013](../desktop/bridge.py#L1013) |
| 9 | Export-CSV-Spalte „Fehlermeldung“ immer leer; kein Prompt-Hash in dieser CSV | [bridge.py:1152](../desktop/bridge.py#L1152) |
| 10 | Rohantwort des Modells wird nicht gespeichert | `run_analysis` |
| 11 | Keine Hash-Duplikatprüfung beim Import von Prüfkörperbildern | [bridge.py:576-597](../desktop/bridge.py#L576-L597) |
| 12 | Klassenänderung im Referenzspeicher verschiebt die Datei nicht | `ReferenceStore.update_entry` |
| 13 | App-Referenzspeicher ist nicht an Modellaufrufe angebunden | Architektur |
| 14 | Batches unterstützen keine Evaluationsvariante | [bridge.py:892-909](../desktop/bridge.py#L892-L909) |
| 15 | Evaluation C/Mistral protokolliert ungekürzte Referenzanzahl | [evaluation.py:566](../Tool/evaluation.py#L566) |
| 16 | Ergebnis-Zuordnung über Dateinamenpräfix | [bridge.py:236](../desktop/bridge.py#L236) |
| 17 | `export_overview` überschreibt dieselbe Datei | [bridge.py:1041](../desktop/bridge.py#L1041) |
| 18 | Kausalitätsfilter nur für wenige deutsche Wortformen | [ki_bildanalyse.py:72-75](../Tool/ki_bildanalyse.py#L72-L75) |

# KI-Bildanalyse 1.0

Native macOS-Oberfläche nach der React/pywebview-Architektur aus Use Case 1. Das vorhandene Modul `Tool/ki_bildanalyse.py` bleibt die einzige Klassifikationslogik.

## Entwicklung starten

```bash
npm --prefix frontend install
npm --prefix frontend run build
python3 app.py
```

Für Frontend-Entwicklung mit Hot Reload:

```bash
npm --prefix frontend run dev
python3 app.py --dev-url http://127.0.0.1:5173
```

## macOS-App bauen

```bash
python3 -m pip install -r requirements-build.txt
chmod +x build_macos.sh
./build_macos.sh
```

Das App-Bundle wird unter `dist/KI-Bildanalyse.app` erstellt. Qwen3-VL und LLaVA verwenden die lokale Ollama-Instanz. OpenAI und Mistral werden nur nach zusätzlicher Kostenbestätigung in der Oberfläche aufgerufen; die CLI verlangt weiterhin `--live --confirm-paid-request`. Der Mistral-Key wird ausschließlich aus dem macOS-Schlüsselbunddienst `Usecase_2_Mistral` gelesen.

Projektmetadaten, Ergebnisse, fachliche Korrekturen und Exporte liegen getrennt von Originalbildern unter `Output/EKI_Bildanalyse/Projekte/`.

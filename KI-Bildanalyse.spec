# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import json
import sys

project_root = Path(SPECPATH)
if sys.platform != "darwin":
    raise RuntimeError("Diese V1-Spec ist für macOS vorgesehen.")

datas = [
    (str(project_root / "frontend/dist"), "frontend/dist"),
    (str(project_root / "assets/logo.png"), "assets"),
    (str(project_root / "assets/widget.png"), "assets"),
    (str(project_root / "Datenbank/klassifikationsregeln.json"), "Datenbank"),
    (str(project_root / "Datenbank/versagensarten_erweitert.json"), "Datenbank"),
    (str(project_root / "Tool/model_verification.json"), "Tool"),
    (str(project_root / "Tool/mistral_model_verification.json"), "Tool"),
]

manifest_path = project_root / "Datenbank/Referenzbilder/referenzbilder_manifest.json"
datas.append((str(manifest_path), "Datenbank/Referenzbilder"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for reference_set in manifest["reference_sets"]:
    copied_path = project_root / reference_set["reference_image"]["copied_path"]
    datas.append((str(copied_path), str(copied_path.parent.relative_to(project_root))))

a = Analysis([str(project_root / "app.py")], pathex=[str(project_root)], binaries=[], datas=datas,
             hiddenimports=["webview", "openai"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="KI-Bildanalyse", console=False,
          icon=str(project_root / "assets/app_icon.icns"), argv_emulation=False)
collect = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="KI-Bildanalyse")
app = BUNDLE(collect, name="KI-Bildanalyse.app", icon=str(project_root / "assets/app_icon.icns"),
             bundle_identifier="at.holzbauforschung.ki-bildanalyse", version="1.0.0",
             info_plist={"CFBundleDisplayName":"KI-Bildanalyse","NSHighResolutionCapable":True,
                         "LSMinimumSystemVersion":"12.0"})

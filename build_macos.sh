#!/bin/zsh

set -euo pipefail

PROJECT_DIR="${0:A:h}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    BUILD_PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1 && python -c "import PyInstaller" 2>/dev/null; then
    BUILD_PYTHON="python"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import PyInstaller" 2>/dev/null; then
    BUILD_PYTHON="python3"
else
    print -u2 "PyInstaller wurde in keinem verfügbaren Python-Interpreter gefunden."
    print -u2 "Installation: python3 -m pip install -r requirements-build.txt"
    exit 1
fi

cd "$PROJECT_DIR"

if ! command -v npm >/dev/null 2>&1; then
    print -u2 "npm wurde nicht gefunden. Für den Frontend-Build wird Node.js mit npm benötigt."
    exit 1
fi

print "Installiere Frontend-Abhängigkeiten reproduzierbar …"
if [[ -f frontend/package-lock.json ]]; then
    npm --prefix frontend ci
else
    npm --prefix frontend install
fi

print "Baue React-Frontend …"
npm --prefix frontend run build

for resource in \
    assets/logo.png \
    assets/widget.png \
    assets/app_icon.icns \
    frontend/dist/index.html \
    Datenbank/klassifikationsregeln.json \
    Datenbank/versagensarten_erweitert.json \
    Datenbank/Referenzbilder/referenzbilder_manifest.json \
    Tool/model_verification.json \
    Tool/mistral_model_verification.json
do
    if [[ ! -f "$resource" ]]; then
        print -u2 "Fehlende Ressource: $resource"
        exit 1
    fi
done

find frontend/dist assets -name '._*' -type f -delete
if command -v xattr >/dev/null 2>&1; then
    xattr -cr frontend/dist assets
fi

"$BUILD_PYTHON" -m PyInstaller --noconfirm --clean KI-Bildanalyse.spec

APP_PATH="$PROJECT_DIR/dist/KI-Bildanalyse.app"
[[ -d "$APP_PATH" ]] || { print -u2 "App-Bundle fehlt: $APP_PATH"; exit 1; }

SIGN_STAGE_DIR="$(mktemp -d /tmp/ki-bildanalyse-sign.XXXXXX)"
SIGN_STAGE_APP="$SIGN_STAGE_DIR/KI-Bildanalyse.app"
trap 'rm -rf "$SIGN_STAGE_DIR"' EXIT

ditto --noextattr --noqtn "$APP_PATH" "$SIGN_STAGE_APP"
find "$SIGN_STAGE_APP" -name '._*' -type f -delete
if command -v xattr >/dev/null 2>&1; then
    find "$SIGN_STAGE_APP" -exec xattr -c {} \; 2>/dev/null || true
fi
codesign --force --deep --sign - "$SIGN_STAGE_APP"
codesign --verify --deep --strict --verbose=2 "$SIGN_STAGE_APP"

mv "$APP_PATH" "$SIGN_STAGE_DIR/unsigned.app"
mv "$SIGN_STAGE_APP" "$APP_PATH"
# Der signierte Stand wurde bereits im neutralen Verzeichnis streng geprüft.
# Im synchronisierten Desktop-Projekt hängt FileProvider am Ziel unmittelbar
# wieder FinderInfo an; eine zweite Prüfung dort wäre deshalb irreführend. Bei
# der Installation wird ohne erweiterte Attribute kopiert und erneut geprüft.

print "App erstellt: $APP_PATH"

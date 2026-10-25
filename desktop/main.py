from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import webview

from desktop.bridge import DesktopBridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLASH_MINIMUM_SECONDS = 7.0


def resource_path(relative: str | Path) -> Path:
    relative = Path(relative)
    bases: list[Path] = []
    if getattr(sys, "frozen", False):
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle:
            bases.append(Path(bundle))
        executable = Path(sys.executable).resolve()
        if executable.parent.name == "MacOS":
            contents = executable.parent.parent
            bases.extend((contents / "Resources", contents / "Frameworks"))
    bases.append(PROJECT_ROOT)
    for base in bases:
        candidate = base / relative
        if candidate.exists():
            return candidate
    return bases[0] / relative


def frontend_url(dev_url: str | None = None) -> str:
    if dev_url:
        return dev_url
    configured = os.getenv("EKI_FRONTEND_URL", "").strip()
    if configured:
        return configured
    index = resource_path("frontend/dist/index.html")
    if not index.is_file():
        raise FileNotFoundError("frontend/dist fehlt. Bitte `npm --prefix frontend run build` ausführen.")
    return index.as_uri()


def splash_html() -> str:
    logo = base64.b64encode(resource_path("assets/widget.png").read_bytes()).decode()
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><style>
  :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
  * {{ box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
  body {{ display: grid; place-content: center; background: #fff; color: #082f42; text-align: center; }}
  main {{ width: 640px; height: 390px; padding: 48px 72px 42px; border: 1px solid #dbe5ea; }}
  img {{ display: block; width: 150px; height: auto; margin: 0 auto 24px; }}
  h1 {{ margin: 0; font-size: 27px; line-height: 1.2; letter-spacing: -.3px; }}
  p {{ margin: 6px 0 28px; color: #66808b; font-size: 14px; }}
  span {{ display: block; margin-bottom: 11px; color: #66808b; font-size: 12px; }}
  .track {{ position: relative; width: 400px; height: 8px; margin: auto; overflow: hidden;
            border-radius: 999px; background: #dcebef; }}
  i {{ position: absolute; inset: 0 auto 0 -35%; width: 35%; border-radius: inherit;
       background: #07516b; animation: progress 1.15s ease-in-out infinite; }}
  @keyframes progress {{ to {{ left: 100%; }} }}
</style></head><body><main>
  <img src="data:image/png;base64,{logo}" alt="KI-Bildanalyse">
  <h1>KI-Bildanalyse</h1>
  <p>KI-gestützte Analyse von Holzprüfkörpern</p>
  <span>KI-Bildanalyse wird gestartet …</span>
  <div class="track"><i></i></div>
</main></body></html>"""


def launch(dev_url: str | None = None) -> None:
    bridge = DesktopBridge()
    splash = webview.create_window("KI-Bildanalyse", html=splash_html(), width=640, height=390,
                                   min_size=(640, 390), resizable=False, frameless=True,
                                   easy_drag=True, on_top=True, background_color="#FFFFFF")
    main_window = webview.create_window("KI-Bildanalyse", frontend_url(dev_url), js_api=bridge,
                                        min_size=(1100, 700), hidden=True,
                                        background_color="#F3F4F6")

    def ready() -> None:
        splash.events.loaded.wait()
        shown = time.monotonic()
        main_window.events.loaded.wait()
        deadline = time.monotonic() + 10
        while True:
            try:
                app_ready = bool(main_window.evaluate_js("Boolean(window.__EKI_READY__)"))
            except Exception:
                app_ready = False
            if app_ready or time.monotonic() >= deadline:
                if not app_ready:
                    try:
                        startup_error = main_window.evaluate_js("window.__EKI_ERROR__ || ''")
                    except Exception:
                        startup_error = ""
                    print("Warnung: React-Bereitschaft nicht rechtzeitig bestätigt.",
                          file=sys.stderr, flush=True)
                    if startup_error:
                        print(f"Frontend-Fehler: {startup_error}", file=sys.stderr, flush=True)
                break
            time.sleep(0.05)
        time.sleep(max(0, SPLASH_MINIMUM_SECONDS - (time.monotonic() - shown)))
        splash.hide()
        main_window.show()
        main_window.maximize()

    main_window.events.closed += lambda: splash.destroy()
    webview.start(ready, debug=bool(dev_url or os.getenv("EKI_WEB_DEBUG")))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-url")
    args = parser.parse_args(argv)
    launch(args.dev_url)


if __name__ == "__main__":
    main()

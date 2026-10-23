"""Sichere Mistral-Konfiguration ausschließlich über den macOS-Schlüsselbund."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


KEYCHAIN_SERVICE = "Usecase_2_Mistral"


class MistralConfigurationError(RuntimeError):
    """Die lokale Mistral-Konfiguration ist nicht verwendbar."""


class MistralRequestError(RuntimeError):
    """Ein Mistral-Aufruf ist fehlgeschlagen, ohne sensible Daten offenzulegen."""


@dataclass(frozen=True)
class MistralAPIKeyResolution:
    """Schlüsselbund-Ergebnis; der geheime Wert bleibt in repr ausgeblendet."""

    value: str = field(repr=False)
    source: str = "macos_keychain"
    service: str = KEYCHAIN_SERVICE


def resolve_mistral_api_key(
    *,
    platform: str | None = None,
    keychain_command: Sequence[str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> MistralAPIKeyResolution:
    """Liest den Mistral-Key ausschließlich aus dem macOS-Schlüsselbund."""

    current_platform = sys.platform if platform is None else platform
    if current_platform != "darwin":
        raise MistralConfigurationError(
            "Mistral ist für Use Case 2 ausschließlich über den macOS-Schlüsselbund konfiguriert."
        )
    command = list(
        keychain_command
        or ("security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w")
    )
    try:
        completed = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        key = completed.stdout.strip()
        if key:
            return MistralAPIKeyResolution(key)
    except (OSError, subprocess.SubprocessError):
        pass
    raise MistralConfigurationError(
        f"Kein Mistral-Schlüssel im macOS-Schlüsselbunddienst {KEYCHAIN_SERVICE!r} gefunden."
    )


_SENSITIVE_ERROR_KEYS = {
    "api_key", "apikey", "authorization", "body", "data", "image", "image_url",
    "images", "input", "payload", "request", "token",
}
_DATA_URL = re.compile(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_BEARER_TOKEN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+\-/=]+")


def _redact_mistral_error_value(value: Any, *, depth: int = 0) -> Any:
    """Behält Validierungsdetails, entfernt aber Request-, Schlüssel- und Bildinhalte."""

    if depth > 12:
        return "[zu tief verschachtelt]"
    if isinstance(value, dict):
        return {
            str(key): (
                "[geschwärzt]"
                if str(key).casefold() in _SENSITIVE_ERROR_KEYS
                else _redact_mistral_error_value(item, depth=depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_mistral_error_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        redacted = _DATA_URL.sub("[Bildinhalt geschwärzt]", value)
        redacted = _BEARER_TOKEN.sub("Bearer [geschwärzt]", redacted)
        return redacted if len(redacted) <= 2000 else redacted[:2000] + "…[gekürzt]"
    return value


def _mistral_http_error_details(exc: Exception) -> str | None:
    """Liest einen HTTP-Fehlerbody vollständig und gibt nur sichere Details zurück."""

    reader = getattr(exc, "read", None)
    if not callable(reader):
        return None
    try:
        body = reader()
    except Exception:
        return "HTTP-Antwort konnte nicht gelesen werden"
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not isinstance(body, str) or not body.strip():
        return "HTTP-Antwort enthielt keinen Body"
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        safe_text = _redact_mistral_error_value(body.strip())
        return f"HTTP-Body={safe_text}"
    safe_body = _redact_mistral_error_value(parsed)
    return "HTTP-Body=" + json.dumps(safe_body, ensure_ascii=False, separators=(",", ":"))


def safe_mistral_request_error(exc: Exception, *, model: str) -> MistralRequestError:
    """Erzeugt eine diagnostische Fehlermeldung ohne Schlüssel oder Bildinhalte."""

    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    details = [f"Typ={type(exc).__name__}", f"Modell={model}"]
    if status_code is not None:
        details.append(f"HTTP-Status={status_code}")
    headers = getattr(exc, "headers", None)
    request_id = headers.get("x-request-id") if headers is not None else None
    if request_id:
        details.append(f"Request-ID={request_id}")
    response_details = _mistral_http_error_details(exc)
    if response_details:
        details.append(response_details)
    return MistralRequestError(
        "Mistral-Bildanalyse fehlgeschlagen (" + ", ".join(details) + ")."
    )

"""Sichere OpenAI-Konfiguration ohne Speicherung oder Ausgabe von Schluesseln."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


KEYCHAIN_SERVICE = "KI-Bemessungstool OpenAI"


class OpenAIConfigurationError(RuntimeError):
    """Die lokale OpenAI-Konfiguration ist nicht verwendbar."""


class OpenAIRequestError(RuntimeError):
    """Ein OpenAI-Aufruf ist fehlgeschlagen, ohne sensible Daten offenzulegen."""


@dataclass(frozen=True)
class APIKeyResolution:
    """Ergebnis der Schluesselaufloesung; der Wert wird in repr ausgeblendet."""

    value: str = field(repr=False)
    source: str


def resolve_api_key(
    *,
    environment: Mapping[str, str] | None = None,
    platform: str | None = None,
    keychain_command: Sequence[str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> APIKeyResolution:
    """Loest den Key wie Use Case 1 auf: Umgebung, dann macOS-Schluesselbund."""

    env = os.environ if environment is None else environment
    environment_key = env.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return APIKeyResolution(environment_key, "environment")

    current_platform = sys.platform if platform is None else platform
    if current_platform == "darwin":
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
                return APIKeyResolution(key, "macos_keychain")
        except (OSError, subprocess.SubprocessError):
            pass

    raise OpenAIConfigurationError(
        "Kein OpenAI-Zugang gefunden. Erwartet wird OPENAI_API_KEY oder der "
        f"macOS-Schluesselbunddienst {KEYCHAIN_SERVICE!r}."
    )


def create_openai_client(resolution: APIKeyResolution):
    """Erzeugt den SDK-Client, ohne einen Netzwerkaufruf auszufuehren."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIConfigurationError(
            "Das Python-Paket 'openai' fehlt. Installiere Tool/requirements.txt."
        ) from exc
    return OpenAI(api_key=resolution.value)


def safe_request_error(exc: Exception, *, model: str) -> OpenAIRequestError:
    """Erzeugt eine diagnostische, aber schluesselfreie Fehlermeldung."""

    request_id = getattr(exc, "request_id", None)
    status_code = getattr(exc, "status_code", None)
    details = [f"Typ={type(exc).__name__}", f"Modell={model}"]
    if status_code is not None:
        details.append(f"HTTP-Status={status_code}")
    if request_id:
        details.append(f"Request-ID={request_id}")
    return OpenAIRequestError("OpenAI-Bildanalyse fehlgeschlagen (" + ", ".join(details) + ").")


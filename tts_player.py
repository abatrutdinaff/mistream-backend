"""Utilities for requesting TTS audio and playing the resulting WAV files."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Iterable, List

try:
    import winsound
except ImportError:  # pragma: no cover - winsound доступен только на Windows
    winsound = None  # type: ignore[assignment]

# === Настройки, могут быть переопределены переменными окружения ===
TTS_ENDPOINT_URL: str = os.environ.get(
    "VNREADER_TTS_ENDPOINT",
    "http://localhost:8084/generate_tts",
)
TTS_DEFAULT_SPEAKER: str = os.environ.get("VNREADER_TTS_SPEAKER", "kseniya")
TTS_OUTPUT_BASE_PATH: str = os.environ.get(
    "VNREADER_TTS_OUTPUT_BASE",
    r"f:\\Scripts\\NeuroStream\\shared\\tts_outputs",
)
TTS_REQUEST_TIMEOUT: float = float(os.environ.get("VNREADER_TTS_TIMEOUT", "60"))
TTS_REQUEST_RETRIES: int = int(os.environ.get("VNREADER_TTS_RETRIES", "3"))
TTS_REQUEST_RETRY_DELAY: float = float(os.environ.get("VNREADER_TTS_RETRY_DELAY", "1.0"))
TTS_DISABLE_ENV_PROXIES: bool = os.environ.get(
    "VNREADER_TTS_DISABLE_PROXIES",
    "1",
).lower() in {"1", "true", "yes", "on"}
TTS_REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "vnreader-tts-client/1.0",
}

if TTS_REQUEST_RETRIES < 1:
    TTS_REQUEST_RETRIES = 1
if TTS_REQUEST_RETRY_DELAY < 0:
    TTS_REQUEST_RETRY_DELAY = 0.0

_RETRYABLE_HTTP_CODES = {500, 502, 503, 504}

if TTS_DISABLE_ENV_PROXIES:
    _URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
else:
    _URL_OPENER = urllib.request.build_opener()


class TTSRequestError(RuntimeError):
    """Ошибка при обращении к сервису TTS."""


def _build_payload(text: str, speaker: str) -> bytes:
    payload = [
        {
            "speaker": speaker,
            "phrases": [text],
        }
    ]
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _resolve_audio_path(audio_url: str, base_path: str) -> str:
    normalized = audio_url.replace("\\", "/")
    if "/tts_outputs/" in normalized:
        _, relative = normalized.split("/tts_outputs/", 1)
    else:
        relative = normalized.lstrip("/\\")
    relative = relative.lstrip("/\\")
    full_path = os.path.join(base_path, *relative.split("/"))
    return full_path


def _extract_audio_paths(response: dict, base_path: str) -> List[str]:
    results_field = response.get("results")
    if not isinstance(results_field, list):
        raise TTSRequestError("Некорректный ответ сервиса TTS: отсутствует поле 'results'.")

    paths: List[str] = []
    for speaker_chunk in results_field:
        chunk_results = speaker_chunk.get("results") if isinstance(speaker_chunk, dict) else None
        if not isinstance(chunk_results, Iterable):
            continue
        for item in chunk_results:
            if not isinstance(item, dict):
                continue
            audio_url = item.get("audio_url")
            if not isinstance(audio_url, str):
                continue
            paths.append(_resolve_audio_path(audio_url, base_path))

    if not paths:
        raise TTSRequestError("Сервис TTS не вернул аудиофайлы.")

    return paths


def generate_tts_audio(text: str, speaker: str | None = None) -> List[str]:
    """Отправляет текст на TTS сервер и возвращает список путей к WAV файлам."""
    speaker_name = speaker or TTS_DEFAULT_SPEAKER
    if not text.strip():
        raise ValueError("Нельзя отправить пустой текст в TTS.")

    payload = _build_payload(text.strip(), speaker_name)
    response_bytes = _perform_tts_request(payload)

    try:
        response_json = json.loads(response_bytes.decode("utf-8"))
    except Exception as exc:
        raise TTSRequestError("Некорректный JSON в ответе сервиса TTS.") from exc

    return _extract_audio_paths(response_json, TTS_OUTPUT_BASE_PATH)


def _perform_tts_request(payload: bytes) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, TTS_REQUEST_RETRIES + 1):
        request = urllib.request.Request(
            TTS_ENDPOINT_URL,
            data=payload,
            headers=TTS_REQUEST_HEADERS,
            method="POST",
        )
        try:
            with _URL_OPENER.open(request, timeout=TTS_REQUEST_TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            details = body.decode("utf-8", errors="ignore").strip()
            if len(details) > 500:
                details = details[:497] + "..."
            message = f"Сервис TTS вернул HTTP {exc.code}"
            if exc.reason:
                message += f" ({exc.reason})"
            if details:
                message += f": {details}"
            if exc.code in _RETRYABLE_HTTP_CODES and attempt < TTS_REQUEST_RETRIES:
                last_error = TTSRequestError(message)
                time.sleep(TTS_REQUEST_RETRY_DELAY)
                continue
            raise TTSRequestError(message) from exc
        except Exception as exc:
            last_error = exc
            if attempt < TTS_REQUEST_RETRIES:
                time.sleep(TTS_REQUEST_RETRY_DELAY)
                continue
            raise TTSRequestError(f"Не удалось обратиться к сервису TTS: {exc}") from exc

    raise TTSRequestError(
        "Не удалось обратиться к сервису TTS: повторные попытки исчерпаны." if last_error is None
        else f"Не удалось обратиться к сервису TTS: {last_error}"
    )


def play_audio_file(path: str) -> None:
    """Воспроизводит WAV файл синхронно."""
    if winsound is None:  # pragma: no cover - для тестового окружения без Windows
        raise RuntimeError(
            "winsound недоступен в текущем окружении. Воспроизведение невозможно."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Аудиофайл не найден: {path}")

    winsound.PlaySound(path, winsound.SND_FILENAME)


__all__ = [
    "generate_tts_audio",
    "play_audio_file",
    "TTS_ENDPOINT_URL",
    "TTS_DEFAULT_SPEAKER",
    "TTS_OUTPUT_BASE_PATH",
    "TTS_REQUEST_TIMEOUT",
    "TTS_REQUEST_RETRIES",
    "TTS_REQUEST_RETRY_DELAY",
    "TTS_REQUEST_HEADERS",
    "TTS_DISABLE_ENV_PROXIES",
    "TTSRequestError",
]

"""Automation loop for VN reader playback."""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image

from tts_player import TTSRequestError, generate_tts_audio, play_audio_file


@dataclass
class PlayLoopCallbacks:
    """Callbacks used by :class:`PlayLoop` to communicate with the GUI layer."""

    set_status: Callable[[str], None]
    show_warning: Callable[[str, str], None]
    show_error: Callable[[str, str], None]
    on_start: Callable[[], None]
    on_finish: Callable[[], None]


class PlayLoop:
    """Runs the "игра" loop in a background thread."""

    def __init__(
        self,
        ocr_engine: object,
        *,
        grab_window: Callable[[int], Optional[Image.Image]],
        double_click: Callable[[int], None],
        click_to_ocr_delay: float = 0.15,
        repeat_delay: float = 0.2,
        extra_delay_range: tuple[float, float] = (0.12, 0.3),
        stability_interval: float = 0.02,
        stability_required_matches: int = 3,
        stability_min_duration: float = 0.4,
        stability_threshold: float = 1.5,
        stability_max_wait: float = 1.8,
        stability_sample_max_dim: int = 480,
    ) -> None:
        self._ocr_engine = ocr_engine
        self._grab_window = grab_window
        self._double_click = double_click
        # Между двойным кликом и OCR выдерживаем минимум 100 мс.
        self._click_to_ocr_delay = max(0.2, click_to_ocr_delay)
        self._repeat_delay = max(0.0, repeat_delay)
        self._extra_delay_range = (
            max(0.0, extra_delay_range[0]),
            max(extra_delay_range[0], extra_delay_range[1]),
        )
        self._stability_interval = max(0.005, stability_interval)
        self._stability_required = max(1, int(stability_required_matches))
        self._stability_min_duration = max(0.0, stability_min_duration)
        self._stability_threshold = max(0.1, stability_threshold)
        self._stability_max_wait = max(self._stability_interval, stability_max_wait)
        self._stability_sample_dim = max(64, stability_sample_max_dim)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_full_text: str = ""

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, hwnd: int, callbacks: PlayLoopCallbacks) -> bool:
        with self._lock:
            if self.is_running():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="play-loop",
                args=(hwnd, callbacks),
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_running():
                return False
            self._stop_event.set()
            return True

    def _wait(self, seconds: float) -> bool:
        return self._stop_event.wait(seconds)

    def _grab_image(self, hwnd: int, callbacks: PlayLoopCallbacks) -> Optional[Image.Image]:
        try:
            return self._grab_window(hwnd)
        except Exception as exc:  # pragma: no cover - зависит от win32 окружения
            callbacks.show_warning(
                "Ошибка захвата",
                f"Не удалось получить изображение окна: {exc}",
            )
            return None

    def _prepare_array(self, image: Image.Image) -> np.ndarray:
        gray = image.convert("L")
        w, h = gray.size
        max_dim = max(w, h)
        if max_dim > self._stability_sample_dim:
            scale = self._stability_sample_dim / float(max_dim)
            new_size = (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            )
            gray = gray.resize(new_size, Image.BILINEAR)
        return np.asarray(gray, dtype=np.int16)

    def _are_frames_similar(
        self,
        prev_arr: np.ndarray,
        curr_arr: np.ndarray,
    ) -> bool:
        if prev_arr.shape != curr_arr.shape:
            return False
        diff = np.mean(np.abs(curr_arr - prev_arr))
        return diff <= self._stability_threshold

    def _wait_for_stable_frame(
        self,
        hwnd: int,
        callbacks: PlayLoopCallbacks,
        *,
        allow_partial: bool = True,
    ) -> Optional[Tuple[Image.Image, np.ndarray]]:
        image = self._grab_image(hwnd, callbacks)
        if image is None:
            return None
        arr = self._prepare_array(image)
        stable_matches = 0
        stable_start: Optional[float] = None
        start = time.perf_counter()

        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - start
            if elapsed >= self._stability_max_wait:
                return (image, arr) if allow_partial else None
            if self._wait(self._stability_interval):
                return None
            next_image = self._grab_image(hwnd, callbacks)
            if next_image is None:
                continue
            next_arr = self._prepare_array(next_image)
            if self._are_frames_similar(arr, next_arr):
                stable_matches += 1
                if stable_matches == 1:
                    stable_start = time.perf_counter()
                if stable_matches >= self._stability_required:
                    if (
                        stable_start is None
                        or (time.perf_counter() - stable_start)
                        >= self._stability_min_duration
                    ):
                        return next_image, next_arr
            else:
                stable_matches = 0
                stable_start = None
                image, arr = next_image, next_arr
        return None

    def _process_image(
        self,
        image: Image.Image,
        callbacks: PlayLoopCallbacks,
    ) -> Tuple[bool, Optional[str]]:
        text = self._ocr_engine.ocr_image(image).strip()

        print("\n" + "=" * 25 + " PLAY LOOP OCR START " + "=" * 25)
        print(text)
        print("=" * 24 + " PLAY LOOP OCR END " + "=" * 25 + "\n", flush=True)

        if not text:
            callbacks.set_status(
                "Игровой цикл: текст не распознан, повторяем..."
            )
            return False, None

        new_text = self._extract_new_content(text)
        if new_text != text:
            print("[PlayLoop] Trimmed repeated prefix ->", new_text, flush=True)
        if not new_text:
            callbacks.set_status(
                "Игровой цикл: новый текст не обнаружен, ждём..."
            )
            self._last_full_text = text
            return False, None

        callbacks.set_status("Игровой цикл: запрос к TTS...")
        try:
            audio_paths = generate_tts_audio(new_text)
        except TTSRequestError as exc:
            callbacks.show_error("Ошибка TTS", str(exc))
            return False, "Игровой цикл: ошибка TTS."
        except Exception as exc:  # pragma: no cover - непредвиденная ошибка
            callbacks.show_error(
                "Ошибка OCR/TTS",
                f"Неожиданная ошибка при обращении к TTS: {exc}",
            )
            return False, "Игровой цикл: ошибка TTS."

        if not audio_paths:
            callbacks.set_status(
                "Игровой цикл: TTS не вернул аудио, повторяем..."
            )
            return False, None

        main_audio = audio_paths[0]
        filename = os.path.basename(main_audio)
        callbacks.set_status(
            f"Игровой цикл: воспроизведение ({filename})..."
        )
        try:
            play_audio_file(main_audio)
        except Exception as exc:  # pragma: no cover - зависит от winsound
            callbacks.show_error(
                "Ошибка воспроизведения",
                f"Не удалось воспроизвести аудио: {exc}",
            )
            return False, "Игровой цикл: ошибка воспроизведения."

        self._last_full_text = text

        extra_delay = random.uniform(*self._extra_delay_range)
        if self._wait(extra_delay):
            return True, "Игровой цикл: остановлен."
        return True, None

    def _extract_new_content(self, current_text: str) -> str:
        if not self._last_full_text:
            return current_text

        prev = self._last_full_text
        prev_lower = prev.lower()
        curr_lower = current_text.lower()

        max_len = min(len(prev_lower), len(curr_lower))
        overlap = 0
        for length in range(max_len, 0, -1):
            if curr_lower.startswith(prev_lower[-length:]):
                overlap = length
                break

        trimmed = current_text[overlap:]
        return trimmed.lstrip(" \t\n\r.,;:!?-—…\"'()[]{}«»")

    def _run_loop(self, hwnd: int, callbacks: PlayLoopCallbacks) -> None:
        self._last_full_text = ""
        callbacks.on_start()
        final_status = "Игровой цикл: остановлен."
        try:
            callbacks.set_status("Игровой цикл: анализ текущего текста...")
            stable = self._wait_for_stable_frame(hwnd, callbacks, allow_partial=False)
            if stable is not None:
                image, _ = stable
                success, status_override = self._process_image(image, callbacks)
                if status_override is not None:
                    final_status = status_override
                    return
                if not success:
                    if self._wait(self._repeat_delay):
                        final_status = "Игровой цикл: остановлен."
                        return
            while not self._stop_event.is_set():
                callbacks.set_status("Игровой цикл: двойной клик...")
                try:
                    self._double_click(hwnd)
                except Exception as exc:  # pragma: no cover - зависит от win32 окружения
                    callbacks.show_error(
                        "Ошибка взаимодействия",
                        f"Не удалось выполнить двойной клик: {exc}",
                    )
                    final_status = "Игровой цикл: ошибка двойного клика."
                    break

                if self._wait(self._click_to_ocr_delay):
                    break

                callbacks.set_status("Игровой цикл: ожидание стабилизации текста...")
                stable = self._wait_for_stable_frame(hwnd, callbacks, allow_partial=False)
                if stable is None:
                    callbacks.set_status(
                        "Игровой цикл: текст не стабилизировался, пробуем ещё раз..."
                    )
                    if self._wait(self._repeat_delay):
                        break
                    continue

                image, _ = stable
                success, status_override = self._process_image(image, callbacks)
                if status_override is not None:
                    final_status = status_override
                    break
                if not success:
                    if self._wait(self._repeat_delay):
                        break
                    continue
        finally:
            with self._lock:
                self._thread = None
            self._last_full_text = ""
            callbacks.set_status(final_status)
            callbacks.on_finish()
            self._stop_event.clear()


__all__ = ["PlayLoop", "PlayLoopCallbacks"]

# -*- coding: utf-8 -*-
"""
VN Bot GUI v5.4: RapidOCR (ONNXRuntime) c CUDA/TensorRT, узким ROI и сборкой строк.

Основные изменения vs v5.3:
- GPU по умолчанию (TensorRT→CUDA→CPU), включён кеш движков и FP16 (если доступен).
- SUB_BAND_REL_HEIGHT = 0.18 и ограничение ширины ROI до MAX_BAND_WIDTH = 960.
- det_limit_side_len=640, rec_batch_num=16 (на GPU), умеренные значения на CPU.
- Кластеризация по Y и удаление дублей, чтобы строки не превращались в «кашу».
"""

import os
import sys
import time
import threading
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
from difflib import SequenceMatcher

import psutil

# GUI
import tkinter as tk
from tkinter import ttk, messagebox

# Win32
import win32gui
import win32con
import win32api
import win32process
import win32ui

# Imaging / OCR
from PIL import Image
import numpy as np
import cv2
cv2.setNumThreads(0)

# Быстрый захват экрана
try:
    import mss
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

# ==== НАСТРОЙКИ ====
# Включаем GPU по умолчанию — это критично для скорости.
USE_GPU = True

# Порог уверенности символов/строк
MIN_SCORE = 0.45

# Высота нижней полосы (субтитры) относительно всего окна
SUB_BAND_REL_HEIGHT = 0.18     # было 0.30 — делаем уже

# Кэп ширины ROI перед детектором (ускоряет детект)
MAX_BAND_WIDTH = 960

# ===================

RAPID_AVAILABLE = True
try:
    from rapidocr_onnxruntime import RapidOCR
    from huggingface_hub import hf_hub_download
except Exception:
    RAPID_AVAILABLE = False


# ------------------------ Окна и взаимодействие ------------------------

@dataclass
class WindowItem:
    hwnd: int
    pid: int
    title: str
    exe: str

    def display(self) -> str:
        t = (self.title or "").strip()
        if len(t) > 80:
            t = t[:77] + "..."
        return f"[PID {self.pid}] {self.exe} — {t}"


def _is_alt_tab_window(hwnd: int) -> bool:
    if not win32gui.IsWindowVisible(hwnd):
        return False
    if win32gui.GetWindowText(hwnd) == "":
        return False
    if win32gui.GetParent(hwnd) != 0:
        return False
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    if style & win32con.WS_EX_TOOLWINDOW:
        return False
    return True


def enum_top_windows() -> List[WindowItem]:
    result: List[WindowItem] = []

    def callback(hwnd, _):
        if not _is_alt_tab_window(hwnd):
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            title = win32gui.GetWindowText(hwnd)
            exe = "unknown.exe"
            try:
                exe = psutil.Process(pid).name()
            except Exception:
                pass
            result.append(WindowItem(hwnd=hwnd, pid=pid, title=title, exe=exe))
        except Exception:
            pass
        return True

    win32gui.EnumWindows(callback, None)
    result.sort(key=lambda w: (w.exe.lower(), (w.title or "").lower()))
    return result


def bring_window_to_foreground(hwnd: int) -> None:
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        try:
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            win32gui.SetForegroundWindow(hwnd)
        finally:
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.1)


def make_lparam(x: int, y: int) -> int:
    return (y << 16) | x


def post_message_double_click(hwnd: int, delay_ms: int = 50) -> None:
    bring_window_to_foreground(hwnd)
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    lparam = make_lparam(cx, cy)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(0.01)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
    time.sleep(delay_ms / 1000.0)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(0.01)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)


def grab_window_printwindow(hwnd: int) -> Optional[Image.Image]:
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width, height = max(1, right - left), max(1, bottom - top)
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)
        ok = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1)
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr,
                               'raw', 'BGRX', 0, 1)
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return img if ok == 1 else None
    except Exception:
        return None


def get_client_rect_screen(hwnd: int) -> Tuple[int, int, int, int]:
    left_rel, top_rel, right_rel, bottom_rel = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left_rel, top_rel))
    right, bottom = win32gui.ClientToScreen(hwnd, (right_rel, bottom_rel))
    return left, top, right, bottom


def grab_window_screen(hwnd: int) -> Optional[Image.Image]:
    try:
        left, top, right, bottom = get_client_rect_screen(hwnd)
        width, height = max(1, right - left), max(1, bottom - top)
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                bbox = {"left": left, "top": top, "width": width, "height": height}
                sct_img = sct.grab(bbox)
                return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        else:
            from PIL import ImageGrab
            return ImageGrab.grab(bbox=(left, top, right, bottom)).convert("RGB")
    except Exception:
        return None


def grab_window(hwnd: int) -> Optional[Image.Image]:
    bring_window_to_foreground(hwnd)
    return grab_window_printwindow(hwnd) or grab_window_screen(hwnd)


# ------------------------ OCR движок ------------------------

class OCREngine:
    def __init__(self, band_rel_h: float = SUB_BAND_REL_HEIGHT, use_gpu: bool = USE_GPU):
        self.band_rel_h = band_rel_h
        self.use_gpu = use_gpu
        self._ocr = None
        self._lock = threading.Lock()
        self._models = None  # (det, rec, dict)

    # ----- Подготовка моделей -----

    def _ensure_models(self):
        if self._models is not None:
            return
        det = hf_hub_download("monkt/paddleocr-onnx", "detection/v5/det.onnx")
        rec = hf_hub_download("monkt/paddleocr-onnx", "languages/eslav/rec.onnx")
        dic = hf_hub_download("monkt/paddleocr-onnx", "languages/eslav/dict.txt")
        self._models = (det, rec, dic)

    def _ensure_loaded(self):
        if self._ocr is not None:
            return
        self._ensure_models()
        det, rec, dic = self._models

        providers_hint = None
        if self.use_gpu:
            # Настройки для TensorRT/CUDA
            cache_dir = os.path.join(os.path.expanduser("~"), ".ort_trt_cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
            os.environ.setdefault("ORT_TENSORRT_CACHE_PATH", cache_dir)
            os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")  # если GPU поддерживает
            providers_hint = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"
            providers_hint = ["CPUExecutionProvider"]

        # Конструктор RapidOCR поддерживает det_limit_side_len / rec_batch_num в последних версиях.
        kwargs_common = dict(
            det_model_path=det,
            rec_model_path=rec,
            rec_keys_path=dic,
            use_angle_cls=False
        )

        # Быстрые пресеты
        if self.use_gpu:
            kwargs_common.update(dict(
                det_limit_side_len=640,
                rec_batch_num=16,
                det_use_cuda=True,
                cls_use_cuda=True,
                rec_use_cuda=True,
                det_use_dml=False,
                cls_use_dml=False,
                rec_use_dml=False,
            ))
        else:
            kwargs_common.update(dict(det_limit_side_len=640, rec_batch_num=6))

        try:
            self._ocr = RapidOCR(providers=providers_hint, **kwargs_common)
        except TypeError:
            # На случай старых версий без providers/параметров — откатываемся к базовым
            try:
                kwargs_fallback = dict(
                    det_model_path=det,
                    rec_model_path=rec,
                    rec_keys_path=dic,
                    use_angle_cls=False
                )
                if self.use_gpu:
                    kwargs_fallback.update(dict(
                        det_use_cuda=True,
                        cls_use_cuda=True,
                        rec_use_cuda=True,
                        det_use_dml=False,
                        cls_use_dml=False,
                        rec_use_dml=False,
                    ))
                self._ocr = RapidOCR(**kwargs_fallback)
            except Exception:
                if self.use_gpu:
                    try:
                        self._ocr = RapidOCR(
                            det_model_path=det,
                            rec_model_path=rec,
                            rec_keys_path=dic,
                            det_use_cuda=True,
                            cls_use_cuda=True,
                            rec_use_cuda=True,
                            det_use_dml=False,
                            cls_use_dml=False,
                            rec_use_dml=False,
                        )
                    except Exception:
                        self._ocr = RapidOCR(det_model_path=det, rec_model_path=rec, rec_keys_path=dic)
                else:
                    self._ocr = RapidOCR(det_model_path=det, rec_model_path=rec, rec_keys_path=dic)

        # === ИЗМЕНЕНИЕ: Добавлен диагностический блок для определения CPU/GPU ===
        if self._ocr:
            try:
                # Получаем список активных провайдеров из сессий ONNX
                providers_det = []
                providers_rec = []
                providers_cls = []

                try:
                    providers_det = self._ocr.text_det.infer.session.get_providers()
                except AttributeError:
                    providers_det = []

                try:
                    providers_rec = self._ocr.text_rec.infer.session.get_providers()
                except AttributeError:
                    providers_rec = []

                try:
                    providers_cls = self._ocr.text_cls.infer.session.get_providers()
                except AttributeError:
                    providers_cls = []

                all_providers = [
                    ("det", providers_det),
                    ("rec", providers_rec),
                    ("cls", providers_cls),
                ]

                # Определяем режим по любому из провайдеров
                mode = "НЕИЗВЕСТНО"
                flat_providers = [p for _, plist in all_providers for p in plist]
                if flat_providers:
                    if any("CUDA" in p or "Tensorrt" in p for p in flat_providers):
                        mode = "GPU"
                    elif any("CPU" in p for p in flat_providers):
                        mode = "CPU"

                print("\n" + "=" * 60, flush=True)
                print(f"[OCREngine] Движок OCR загружен. Режим: {mode}")
                for name, providers in all_providers:
                    if providers:
                        print(f"  [{name}] providers: {', '.join(providers)}")
                print("=" * 60 + "\n", flush=True)

            except AttributeError:
                # На случай, если в старой версии нет прямого доступа к сессии
                print("\n[OCREngine] Движок OCR загружен. Не удалось автоматически определить провайдер.\n", flush=True)
        # === КОНЕЦ ИЗМЕНЕНИЯ ===


    # ----- Препроцессинг -----

    @staticmethod
    def _enhance_yellow_white(img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (15, 60, 90), (40, 255, 255))
        white  = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
        mask = cv2.bitwise_or(yellow, white)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        boosted = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        boosted = boosted.astype(np.uint8, copy=True)
        m0 = (mask == 0)
        m1 = (mask > 0)
        boosted[m0] = (boosted[m0] * 0.5).astype(np.uint8)
        boosted[m1] = np.clip(boosted[m1] * 1.25, 0, 255).astype(np.uint8)
        return cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)

    def _crop_subtitle_band(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[0] < 2:
            return img_bgr
        h = img_bgr.shape[0]
        y0 = max(0, min(h - 1, int(h * (1.0 - self.band_rel_h))))
        band = img_bgr[y0:h, :]

        # Кэп по ширине для ускорения детектора
        h, w = band.shape[:2]
        if w > MAX_BAND_WIDTH:
            scale = MAX_BAND_WIDTH / float(w)
            band = cv2.resize(band, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return band

    # ----- Утилиты для разбора боксов -----

    @staticmethod
    def _xyw_from_box(box: Any) -> Tuple[float, float, float]:
        """Возвращает (x_left, y_center, width) из бокса RapidOCR"""
        try:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                # ожидается [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
                x0, y0 = float(box[0][0]), float(box[0][1])
                x1, y1 = float(box[1][0]), float(box[1][1])
                x2, y2 = float(box[2][0]), float(box[2][1])
                x3, y3 = float(box[3][0]), float(box[3][1])
                x_left = min(x0, x1, x2, x3)
                y_center = (y0 + y1 + y2 + y3) / 4.0
                width = max(x0, x1, x2, x3) - x_left
                return x_left, y_center, max(1.0, width)
        except Exception:
            pass
        return 0.0, 0.0, 1.0

    # ----- Нормализация вывода RapidOCR -----

    def _normalize_rapidocr_output(self, output: Any) -> List[Tuple[float, float, str, Optional[float]]]:
        """
        -> список (x_left, y_center, text, score)
        """
        items: List[Tuple[float, float, str, Optional[float]]] = []

        try:
            # Вариант: (dt_boxes, rec_res)
            if isinstance(output, tuple) and len(output) >= 2:
                dt_boxes, rec_res = output[0], output[1]
                if isinstance(rec_res, list):
                    for i, rr in enumerate(rec_res):
                        text = None
                        score: Optional[float] = None
                        if isinstance(rr, (list, tuple)):
                            if len(rr) >= 1 and isinstance(rr[0], str):
                                text = rr[0]
                                if len(rr) >= 2 and isinstance(rr[1], (int, float)):
                                    score = float(rr[1])
                        elif isinstance(rr, str):
                            text = rr
                        x_left, y_center, _ = 0.0, 0.0, 1.0
                        if isinstance(dt_boxes, list) and i < len(dt_boxes):
                            x_left, y_center, _ = self._xyw_from_box(dt_boxes[i])
                        if text:
                            items.append((x_left, y_center, text, score))
                    if items:
                        return items

            # Вариант: список элементов
            primary = output[0] if isinstance(output, tuple) and len(output) >= 1 else output
            if isinstance(primary, list):
                for item in primary:
                    text = None
                    score: Optional[float] = None
                    x_left, y_center = 0.0, 0.0

                    if isinstance(item, (list, tuple)):
                        if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                            box, ts = item[0], item[1]
                            if len(ts) >= 1 and isinstance(ts[0], str):
                                text = ts[0]
                                if len(ts) >= 2 and isinstance(ts[1], (int, float)):
                                    score = float(ts[1])
                            x_left, y_center, _ = self._xyw_from_box(box)
                        elif len(item) >= 3 and isinstance(item[1], str):
                            box = item[0]
                            text = item[1]
                            if isinstance(item[2], (int, float)):
                                score = float(item[2])
                            x_left, y_center, _ = self._xyw_from_box(box)
                        elif len(item) >= 1 and isinstance(item[0], str):
                            text = item[0]
                    elif isinstance(item, str):
                        text = item

                    if text:
                        items.append((x_left, y_center, text, score))

            if not items and isinstance(output, str):
                items.append((0.0, 0.0, output, None))
        except Exception:
            pass

        return items

    # ----- Безопасный вызов RapidOCR -----

    def _rapid_ocr_safe(self, img_bgr: np.ndarray) -> List[Tuple[float, float, str, Optional[float]]]:
        try:
            raw = self._ocr(img_bgr)
        except Exception:
            print("[RapidOCR] Exception during inference:", file=sys.stderr, flush=True)
            traceback.print_exc()
            return []
        return self._normalize_rapidocr_output(raw)

    # ----- Сборка строк из боксов -----

    @staticmethod
    def _similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _assemble_lines(self, items: List[Tuple[float, float, str, Optional[float]]]) -> List[str]:
        # Фильтр по score
        filt: List[Tuple[float, float, str]] = []
        for x, y, t, s in items:
            if t and (s is None or s >= MIN_SCORE):
                filt.append((float(x), float(y), t))

        if not filt:
            return []

        # Группируем по строкам по Y (толеранс 24 px)
        filt.sort(key=lambda v: v[1])
        groups: List[Dict[str, Any]] = []
        for x, y, t in filt:
            placed = False
            for g in groups:
                if abs(y - g["y_mean"]) < 24:
                    g["items"].append((x, y, t))
                    g["y_sum"] += y
                    g["n"] += 1
                    g["y_mean"] = g["y_sum"] / g["n"]
                    placed = True
                    break
            if not placed:
                groups.append({"y_mean": y, "y_sum": y, "n": 1, "items": [(x, y, t)]})

        groups.sort(key=lambda g: g["y_mean"])

        lines: List[str] = []
        for g in groups:
            parts = sorted(g["items"], key=lambda v: v[0])  # sort by x
            merged: List[str] = []
            last_text = ""
            last_x = None
            for x, _, t in parts:
                if last_text:
                    sim = self._similar(last_text.lower(), t.lower())
                    if sim > 0.85 and (last_x is not None and abs(x - last_x) < 40):
                        # очень похожий соседний чанк — считаем дублем
                        continue
                merged.append(t)
                last_text = t
                last_x = x
            line = " ".join(merged).strip()
            if line:
                lines.append(line)
        return lines

    # ----- Основной вызов -----

    def ocr_image(self, img_rgb: Image.Image) -> str:
        with self._lock:
            self._ensure_loaded()

        if img_rgb is None:
            return ""

        np_img = np.array(img_rgb)
        if np_img.ndim != 3 or np_img.shape[1] == 0 or np_img.shape[0] == 0:
            return ""
        np_bgr = np_img[:, :, ::-1].copy()

        band = self._crop_subtitle_band(np_bgr)
        if band is None or band.size == 0:
            return ""

        band = self._enhance_yellow_white(band)
        if band is None or band.size == 0:
            return ""

        t0 = time.time()
        items = self._rapid_ocr_safe(band)
        lines = self._assemble_lines(items)
        dt = (time.time() - t0) * 1000.0

        print(f"[OCR] Done in {dt:.1f} ms; lines: {len(lines)}", flush=True)
        return "\n".join(lines).strip()


# ------------------------ GUI-приложение ------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VN Bot GUI v5.4 (GPU/TensorRT)")
        self.geometry("860x540")
        self.resizable(True, True)
        self.windows: List[WindowItem] = []
        self.selected_hwnd: Optional[int] = None
        self._ocr_engine = OCREngine(band_rel_h=SUB_BAND_REL_HEIGHT, use_gpu=USE_GPU)
        self._build_ui()
        self.refresh_window_list()

        # === ИЗМЕНЕНИЕ: Добавлена предзагрузка движка в фоне при старте ===
        # Это вызовет диагностическое сообщение о CPU/GPU сразу при запуске
        threading.Thread(target=self._preload_ocr_engine, daemon=True).start()

    def _preload_ocr_engine(self):
        print("[App] Запускаю предварительную загрузку OCR движка...", flush=True)
        self._ocr_engine._ensure_loaded()
        self.after(0, self._set_status, f"Готово. Режим OCR: {'GPU' if USE_GPU else 'CPU'} (движок загружен)")
        print("[App] OCR движок готов.", flush=True)
    # === КОНЕЦ ИЗМЕНЕНИЯ ===

    def _build_ui(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm_top, text="Выбери окно процесса:").pack(side=tk.LEFT)
        self.btn_refresh = ttk.Button(frm_top, text="Обновить список", command=self.refresh_window_list)
        self.btn_refresh.pack(side=tk.RIGHT)

        frm_mid = ttk.Frame(self)
        frm_mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.listbox = tk.Listbox(frm_mid, height=16, activestyle="dotbox")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_window)
        scrollbar = ttk.Scrollbar(frm_mid, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        frm_bottom = ttk.Frame(self)
        frm_bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.btn_dbl = ttk.Button(frm_bottom, text="Двойной клик (фоновый)", command=self.handle_double_click)
        self.btn_dbl.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_ocr = ttk.Button(frm_bottom, text="Распознавание текста", command=self.handle_ocr)
        self.btn_ocr.pack(side=tk.LEFT, padx=(0, 10))

        mode = "GPU (TRT→CUDA)" if USE_GPU else "CPU"
        self.lbl_status = ttk.Label(self, text=f"Готово. Режим OCR: {mode}")
        self.lbl_status.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _set_buttons_state(self, state: str):
        self.btn_dbl.config(state=state)
        self.btn_ocr.config(state=state)
        self.btn_refresh.config(state=state)

    def refresh_window_list(self):
        self.windows = enum_top_windows()
        self.listbox.delete(0, tk.END)
        for w in self.windows:
            self.listbox.insert(tk.END, w.display())
        self.selected_hwnd = None
        self.lbl_status.config(text=f"Найдено окон: {len(self.windows)} | OCR: {'GPU' if USE_GPU else 'CPU'}")

    def on_select_window(self, _event=None):
        try:
            idxs = self.listbox.curselection()
            if not idxs:
                self.selected_hwnd = None
                return
            self.selected_hwnd = self.windows[idxs[0]].hwnd
            self.lbl_status.config(text=f"Выбрано HWND={self.selected_hwnd}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выбрать окно: {e}")
            self.selected_hwnd = None

    def handle_double_click(self):
        hwnd = self.selected_hwnd
        if not hwnd:
            messagebox.showwarning("Нет окна", "Сначала выбери окно в списке.")
            return
        self.lbl_status.config(text="Двойной клик...")
        self.update_idletasks()
        try:
            post_message_double_click(hwnd, delay_ms=50)
            self.lbl_status.config(text="Двойной клик выполнен (фоново).")
        except Exception as e:
            self.lbl_status.config(text="Ошибка двойного клика.")
            messagebox.showerror("Ошибка", f"Двойной клик не выполнен: {e}")

    def _ocr_worker(self, hwnd: int):
        try:
            self.after(0, self._set_status, "OCR: захват окна...")
            img = grab_window(hwnd)
            if img is None:
                print("[OCR] Не удалось получить скрин окна.", file=sys.stderr, flush=True)
                self.after(0, self._set_status, "OCR: ошибка захвата окна.")
                return

            self.after(0, self._set_status, "OCR: распознавание...")
            text = self._ocr_engine.ocr_image(img)

            print("\n" + "="*25 + " OCR RESULT START " + "="*25)
            print(text)
            print("="*24 + " OCR RESULT END " + "="*25 + "\n", flush=True)

            self.after(0, self._set_status, "OCR: готово (текст в консоли).")
        except Exception as e:
            print(f"[OCR] Критическая ошибка: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            self.after(0, self._set_status, f"OCR: ошибка ({type(e).__name__}).")
        finally:
            self.after(0, self._set_buttons_state, 'normal')

    def _set_status(self, s: str):
        self.lbl_status.config(text=s)

    def handle_ocr(self):
        hwnd = self.selected_hwnd
        if not hwnd:
            messagebox.showwarning("Нет окна", "Сначала выбери окно в списке.")
            return
        self._set_buttons_state('disabled')
        self.lbl_status.config(text="OCR выполняется...")
        self.update_idletasks()
        t = threading.Thread(target=self._ocr_worker, args=(hwnd,), daemon=False)
        t.start()


def main():
    if not RAPID_AVAILABLE:
        messagebox.showerror(
            "Отсутствует зависимость",
            "Библиотеки 'rapidocr-onnxruntime' или 'huggingface_hub' не найдены.\n"
            "Установите зависимости через Poetry."
        )
        return
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
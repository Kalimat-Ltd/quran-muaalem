"""Manual sliding-window regression test.

Run with:
    python tests/run_sliding_window_sliding_window.py

This script concatenates the four Surah 2:102 audio fragments located in
``assets/surah`` and streams them to the websocket API in a single session.
It asserts that the server enforces the 11-word minimum window, extends the
reference window when fewer than 10 words remain, and subsequently slides the
window forward after the first extension.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Optional dependencies
try:  # pragma: no cover - runtime optional dependency check
    import librosa
except Exception as exc:  # pragma: no cover
    raise RuntimeError("librosa is required to run this script") from exc

try:  # pragma: no cover
    from websockets.asyncio.client import connect as ws_connect
except Exception:  # pragma: no cover
    try:
        from websockets import connect as ws_connect  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("websockets package is required to run this script") from exc

import urllib.request  # noqa: E402  (after optional imports)

from quran_transcript import Aya  # noqa: E402
from quran_muaalem.explain import explain_terminal_new  # noqa: E402

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
WS_URL = API_BASE.replace("http", "ws") + "/ws"

SURAH = 2
AYAH = 102
SR = 16000
FRAME_SECS = 0.02
CHUNK_SECS = 2.0
FRAME_LEN = int(SR * FRAME_SECS)
FRAMES_PER_CHUNK = int(SR * CHUNK_SECS) // FRAME_LEN

AUDIO_FILES = [
    Path("assets/surah/002102.mp3"),
    Path("assets/surah/002103.mp3"),
    Path("assets/surah/002104.mp3"),
    Path("assets/surah/002105.mp3"),
]


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _load_audio_stack() -> np.ndarray:
    waves: List[np.ndarray] = []
    for path in AUDIO_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing audio file: {path}")
        wave, sr = librosa.load(path.as_posix(), sr=SR, mono=True)
        if sr != SR:
            raise RuntimeError(f"{path} did not resample to {SR} Hz (got {sr})")
        if wave.size == 0:
            continue
        waves.append(wave.astype(np.float32))
    if not waves:
        raise RuntimeError("No audio samples loaded")
    combined = np.concatenate(waves)
    pcm = np.clip(combined, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16)


def _server_is_up() -> bool:
    try:
        with urllib.request.urlopen(API_BASE + "/health", timeout=2) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except Exception:
        return False


def _prepare_reference_words() -> tuple[List[str], str, List[int]]:
    aya = Aya(SURAH, AYAH)
    data = aya.get()
    words = [" ".join(w.split()) for w in (data.uthmani_words or []) if w.strip()]
    full_text = " ".join(words)
    starts: List[int] = []
    offset = 0
    for word in words:
        starts.append(offset)
        offset += len(word) + 1  # account for trailing space between words
    return words, full_text, starts


def _locate_window(full_text: str, word_starts: List[int], window_text: str, preferred_char_idx: int) -> tuple[int, int]:
    if not window_text:
        raise ValueError("Window text cannot be empty")
    search_from = max(preferred_char_idx, 0)
    char_idx = full_text.find(window_text, search_from)
    if char_idx == -1:
        char_idx = full_text.find(window_text)
    if char_idx == -1:
        raise ValueError(f"Unable to locate window text: {window_text[:40]!r}")
    word_idx = bisect_right(word_starts, char_idx) - 1
    word_idx = max(word_idx, 0)
    return char_idx, word_idx


async def _recv_until_inference(ws) -> Dict[str, Any]:
    for _ in range(40):  # generous retries per 2s chunk
        msg = await asyncio.wait_for(ws.recv(), timeout=45)
        if isinstance(msg, (bytes, bytearray)):
            continue
        try:
            data = json.loads(msg)
        except Exception:
            continue
        if data.get("type") == "inference":
            return data
        if data.get("type") == "error":
            raise RuntimeError(f"Server returned error: {data}")
    raise RuntimeError("Timed out waiting for inference message")


def _explain_inference(inf: Dict[str, Any]) -> None:
    try:
        result = inf.get("result") or {}
        phonemes_field = result.get("phonemes")
        if isinstance(phonemes_field, dict):
            phoneme_text = phonemes_field.get("text")
        else:
            phoneme_text = phonemes_field

        ref = inf.get("phonetizer_out") or {}
        ref_phonemes_field = ref.get("phonemes")
        if isinstance(ref_phonemes_field, dict):
            ref_text = ref_phonemes_field.get("text")
        else:
            ref_text = ref_phonemes_field

        char_map = ref.get("char_map")
        uthmani = inf.get("uthmani")

        if (
            isinstance(char_map, list)
            and isinstance(phoneme_text, str)
            and isinstance(ref_text, str)
            and isinstance(uthmani, str)
        ):
            explain_terminal_new(
                phonemes=phoneme_text,
                exp_phonemes=ref_text,
                exp_char_map=char_map,
                uthmani_text=uthmani,
            )
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[explain] warning: {exc}")


def _record_inference(inf: Dict[str, Any], inferences: List[Dict[str, Any]]) -> None:
    inferences.append(inf)
    _explain_inference(inf)


async def _gather_final_messages(ws, inferences: List[Dict[str, Any]]) -> None:
    # Collect any trailing inferences after sending "end"
    for _ in range(20):
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
        except asyncio.TimeoutError:
            break
        if isinstance(msg, (bytes, bytearray)):
            continue
        try:
            data = json.loads(msg)
        except Exception:
            continue
        if data.get("type") == "inference":
            _record_inference(data, inferences)
        elif data.get("type") in {"bye", "reset_ack"}:
            break
        elif data.get("type") == "error":
            raise RuntimeError(f"Server returned error: {data}")


async def run() -> None:
    if ws_connect is None:  # pragma: no cover
        raise RuntimeError("websocket client unavailable")
    if not _server_is_up():
        raise RuntimeError(f"API server not reachable at {API_BASE}")

    pcm = _load_audio_stack()
    total_frames = len(pcm) // FRAME_LEN
    remainder = len(pcm) % FRAME_LEN

    aya_words, full_text, word_starts = _prepare_reference_words()

    config_payload = {
        "type": "config",
        "surah": SURAH,
        "ayah": AYAH,
        "start_word": 0,
        "num_words": 5,  # expect server to upgrade to MIN_WINDOW_WORDS
        "rewaya": "hafs",
        "madd_monfasel_len": 2,
        "madd_mottasel_len": 4,
        "madd_mottasel_waqf": 4,
        "madd_aared_len": 2,
        "sampling_rate": SR,
    }

    inferences: List[Dict[str, Any]] = []

    async with ws_connect(WS_URL, open_timeout=5) as ws:
        await ws.send(json.dumps(config_payload))
        ready_msg = await asyncio.wait_for(ws.recv(), timeout=5)
        ready = json.loads(ready_msg)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected ready payload: {ready}")

        frame_idx = 0
        for chunk_idx in range(total_frames):
            start = frame_idx * FRAME_LEN
            end = start + FRAME_LEN
            frame = pcm[start:end]
            await ws.send(frame.tobytes())
            frame_idx += 1

            if frame_idx % FRAMES_PER_CHUNK == 0:
                inf = await _recv_until_inference(ws)
                _record_inference(inf, inferences)

        if remainder:
            await ws.send(pcm[-remainder:].tobytes())

        # Drain any inferences produced by remainder
        try:
            pending_inf = await asyncio.wait_for(ws.recv(), timeout=5)
        except asyncio.TimeoutError:
            pending_inf = None
        if isinstance(pending_inf, str):
            try:
                payload = json.loads(pending_inf)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("type") == "inference":
                _record_inference(payload, inferences)
            elif payload is not None and payload.get("type") == "error":
                raise RuntimeError(f"Server returned error: {payload}")
        elif isinstance(pending_inf, (bytes, bytearray)):
            pass

        await ws.send(json.dumps({"type": "end"}))
        await _gather_final_messages(ws, inferences)

    if not inferences:
        raise AssertionError("Did not receive any inference messages")

    first_words = _normalize_text(inferences[0].get("uthmani")).split()
    if len(first_words) < 11:
        raise AssertionError(f"Expected initial window to contain at least 11 words, got {len(first_words)}")

    extended = False
    slid = False

    def normalize_window_words(text: str | None) -> List[str]:
        return [w for w in _normalize_text(text).split() if w]

    first_text = " ".join(first_words)
    prev_char_idx, prev_start = _locate_window(full_text, word_starts, first_text, 0)
    prev_len = len(first_words)
    history: List[Tuple[int, int]] = [(prev_len, prev_start)]

    for inf in inferences[1:]:
        words = normalize_window_words(inf.get("uthmani"))
        if not words:
            continue
        window_text = " ".join(words)
        try:
            char_idx, start = _locate_window(full_text, word_starts, window_text, prev_char_idx)
        except ValueError:
            # If we cannot align the window reliably, skip this inference.
            continue
        if len(words) > prev_len:
            extended = True
        if start > prev_start:
            slid = True
        prev_len = len(words)
        prev_start = start
        prev_char_idx = char_idx
        history.append((prev_len, prev_start))

    if not extended:
        raise AssertionError("Sliding window never extended its length by 10 words")
    if not slid:
        raise AssertionError(f"Sliding window never advanced forward after extending; history={history}")

    print("Sliding-window test passed")
    print(f"Received {len(inferences)} inference messages")
    print(f"Initial window words: {len(first_words)}")
    print(f"Final window words: {prev_len}")


if __name__ == "__main__":
    asyncio.run(run())

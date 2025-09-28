import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from quran_muaalem.explain import explain_terminal_new

# Optional dependency: websockets
try:
    from websockets.asyncio.client import connect as ws_connect  # websockets>=11
except Exception:  # pragma: no cover
    try:
        from websockets import connect as ws_connect  # type: ignore
    except Exception:
        ws_connect = None  # type: ignore

# Optional dependency: librosa for loading the WAV test file
try:
    from librosa.core import load as librosa_load
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False


API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
WS_URL = API_BASE.replace("http", "ws") + "/ws"

# Test audio and config matching:
# uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani
# moshaf = MoshafAttributes(rewaya="hafs", madd_monfasel_len=2, madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=2)
SURAH = 8
AYAH = 75
START_WORD = 17
NUM_WORDS = 9
REWAYA = "hafs"
MADD_MONFASEL = 2
MADD_MOTTASEL = 4
MADD_MOTTASEL_WAQF = 4
MADD_AARED = 2
SR = 16000


def _print_response(label: str, msg: Any) -> None:
    """Pretty-print server responses without dumping huge arrays."""
    try:
        if isinstance(msg, (bytes, bytearray)):
            print(f"[{label}] <binary> len={len(msg)}")
            return
        # Try parse JSON
        data = json.loads(msg)
        mtype = data.get("type")
        if mtype == "inference":
            wc = data.get("window_chunks")
            ts = data.get("total_samples")
            phon = (data.get("result") or {}).get("phonemes") or {}
            text = phon.get("text") if isinstance(phon, dict) else None
            text_preview = (text[:120] + "…") if isinstance(text, str) and len(text) > 120 else text
            sifat = (data.get("result") or {}).get("sifat") or []
            probs = phon.get("probs") if isinstance(phon, dict) else None
            ids = phon.get("ids") if isinstance(phon, dict) else None
            probs_info = f"list(len={len(probs)})" if isinstance(probs, list) else ("array" if probs is not None else "None")
            ids_info = f"list(len={len(ids)})" if isinstance(ids, list) else ("array" if ids is not None else "None")

            # Summarize phonetizer_out (reference)
            ref = data.get("phonetizer_out")
            ref_text_preview = None
            ref_sifat_count = None
            if isinstance(ref, dict):
                ref_phon = ref.get("phonemes")
                if isinstance(ref_phon, dict):
                    rtext = ref_phon.get("text")
                elif isinstance(ref_phon, str):
                    rtext = ref_phon
                elif isinstance(ref_phon, list):
                    # join first few items if list of strings
                    try:
                        rtext = "".join(ref_phon[:30]) if ref_phon and isinstance(ref_phon[0], str) else None
                    except Exception:
                        rtext = None
                else:
                    rtext = None
                if isinstance(rtext, str):
                    ref_text_preview = (rtext[:120] + "…") if len(rtext) > 120 else rtext
                ref_sifat = ref.get("sifat")
                if isinstance(ref_sifat, list):
                    ref_sifat_count = len(ref_sifat)

            print(
                f"[{label}] type=inference window_chunks={wc} total_samples={ts} "
                f"text={text_preview!r} sifat_count={len(sifat)} probs={probs_info} ids={ids_info} "
                f"ref_text={ref_text_preview!r} ref_sifat_count={ref_sifat_count}"
            )
            explain_terminal_new(phonemes=text or "", exp_phonemes=ref_text_preview, exp_char_map=ref["char_map"], uthmani_text=data.get("uthmani") or "")
        else:
            # Print compact JSON for control messages
            compact = json.dumps(data, ensure_ascii=False)
            if len(compact) > 1000:
                compact = compact[:1000] + "…"
            print(f"[{label}] {compact}")
    except Exception:
        # Not JSON text
        text = str(msg)
        if len(text) > 1000:
            text = text[:1000] + "…"
        print(f"[{label}] TEXT {text}")


def _server_is_up() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(API_BASE + "/health", timeout=2) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except Exception:
        return False


def _wav_path() -> Path:
    # Resolve assets/test.wav from repo root irrespective of cwd
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "assets" / "test.wav"


def _load_wav_pcm16() -> Tuple[np.ndarray, bytes]:
    """Load audio as mono 16kHz and return (int16_samples, pcm_bytes)."""
    wav_path = _wav_path()
    if not wav_path.is_file():
        pytest.skip(f"Missing test audio file: {wav_path}")
    if not HAVE_LIBROSA:
        pytest.skip("librosa is required to load test WAV")

    # librosa returns float32 in [-1,1]
    wave, _ = librosa_load(str(wav_path), sr=SR, mono=True)
    wave = np.asarray(wave, dtype=np.float32)
    # Convert to PCM16 little-endian
    i16 = np.clip(wave, -1.0, 1.0)
    i16 = (i16 * 32767.0).astype(np.int16)
    return i16, i16.tobytes()


@pytest.mark.skipif(ws_connect is None, reason="websockets package not installed")
@pytest.mark.skipif(not _server_is_up(), reason="API server not running on localhost:8000")
@pytest.mark.asyncio
@pytest.mark.slow
async def test_ws_inference_end_to_end():
    # Load test audio once
    i16, pcm = _load_wav_pcm16()

    async with ws_connect(WS_URL, open_timeout=5) as ws:
        # 1) Send config for Aya(8,75) words 17..25 (9 words) with given moshaf
        cfg: Dict[str, Any] = {
            "type": "config",
            "surah": SURAH,
            "ayah": AYAH,
            "start_word": START_WORD,
            "num_words": NUM_WORDS,
            "rewaya": REWAYA,
            "madd_monfasel_len": MADD_MONFASEL,
            "madd_mottasel_len": MADD_MOTTASEL,
            "madd_mottasel_waqf": MADD_MOTTASEL_WAQF,
            "madd_aared_len": MADD_AARED,
            "sampling_rate": SR,
        }
        await ws.send(json.dumps(cfg))

        # Expect ready
        ready_msg = await asyncio.wait_for(ws.recv(), timeout=5)
        _print_response("RECV", ready_msg)
        assert isinstance(ready_msg, str)
        ready = json.loads(ready_msg)
        assert ready.get("type") == "ready"
        assert ready.get("sampling_rate") == SR

        # 2) Ping/Pong
        await ws.send(json.dumps({"type": "ping"}))
        pong_msg = await asyncio.wait_for(ws.recv(), timeout=5)
        _print_response("RECV", pong_msg)
        assert isinstance(pong_msg, str)
        pong = json.loads(pong_msg)
        assert pong.get("type") == "pong"

        # 3) Stream the WAV in 20ms frames (320 samples @ 16kHz). Expect 1 inference per 2s (100 frames)
        frame_len = int(SR * 0.02)  # 20ms -> 320 samples
        two_sec = SR * 2
        frames_per_chunk = two_sec // frame_len  # 100
        total_frames = len(i16) // frame_len
        available_chunks = total_frames // frames_per_chunk
        max_chunks = min(available_chunks, 4)  # collect up to 4 inferences
        assert max_chunks >= 1, "Audio too short for a 2s chunk"

        inferences = []
        frame_idx = 0
        for chunk_idx in range(max_chunks):
            # Send frames for one chunk (2s)
            for _ in range(frames_per_chunk):
                start = frame_idx * frame_len
                end = start + frame_len
                seg = i16[start:end]
                if seg.size == 0:
                    break
                await ws.send(seg.tobytes())
                frame_idx += 1

            # Wait for an inference after completing 2s worth of frames
            inf = None
            for _ in range(20):
                msg = await asyncio.wait_for(ws.recv(), timeout=45)
                _print_response("RECV", msg)
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("type") == "inference":
                    inf = data
                    break
                elif data.get("type") == "error":
                    pytest.fail(f"Server returned error: {data}")
            assert inf is not None, f"No inference received after frames for chunk {chunk_idx+1}"
            assert isinstance(inf.get("window_chunks"), int)
            assert inf["window_chunks"] == chunk_idx + 1

            # Assert phonetizer_out presence and basic structure
            assert "phonetizer_out" in inf
            ref = inf["phonetizer_out"]
            assert isinstance(ref, dict)
            assert "phonemes" in ref
            assert "sifat" in ref

            inferences.append(inf)

        assert len(inferences) == max_chunks

        # 4) Close session
        await ws.send(json.dumps({"type": "end"}))


@pytest.mark.skipif(ws_connect is None, reason="websockets package not installed")
@pytest.mark.skipif(not _server_is_up(), reason=f"API server not running on {WS_URL}")
@pytest.mark.asyncio
async def test_ws_reset_and_reuse():
    i16, pcm = _load_wav_pcm16()

    async with ws_connect(WS_URL, open_timeout=5) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "surah": SURAH,
            "ayah": AYAH,
            "start_word": START_WORD,
            "num_words": NUM_WORDS,
            "rewaya": REWAYA,
            "madd_monfasel_len": MADD_MONFASEL,
            "madd_mottasel_len": MADD_MOTTASEL,
            "madd_mottasel_waqf": MADD_MOTTASEL_WAQF,
            "madd_aared_len": MADD_AARED,
            "sampling_rate": SR,
            "audio_format": "pcm16le",
        }))
        ready_msg = await asyncio.wait_for(ws.recv(), timeout=5)  # ready
        _print_response("RECV", ready_msg)

        # Send a few 20ms frames (< 2s) so no inference should be produced yet
        frame_len = int(SR * 0.02)
        for _ in range(10):  # ~200ms total
            await ws.send(i16[:frame_len].tobytes())

        # Reset buffer
        await ws.send(json.dumps({"type": "reset"}))
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        _print_response("RECV", msg)
        data = json.loads(msg)
        assert data.get("type") == "reset_ack"

        # Now send 2s worth of 20ms frames to trigger inference
        frames_per_chunk = (SR * 2) // frame_len
        offset = 0
        for _ in range(frames_per_chunk):
            start = offset
            end = start + frame_len
            await ws.send(i16[start:end].tobytes())
            offset += frame_len

        # Expect inference
        got_inference = False
        last_inf = None
        for _ in range(20):
            msg = await asyncio.wait_for(ws.recv(), timeout=45)
            _print_response("RECV", msg)  # Print all responses
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "inference":
                got_inference = True
                last_inf = data
                break
            elif data.get("type") == "error":
                pytest.fail(f"Server returned error: {data}")
        assert got_inference, "No inference received after reset"

        # Assert phonetizer_out structure on the received inference
        assert last_inf is not None and "phonetizer_out" in last_inf
        ref = last_inf["phonetizer_out"]
        assert isinstance(ref, dict)
        assert "phonemes" in ref
        assert "sifat" in ref

        await ws.send(json.dumps({"type": "end"}))


if __name__ == "__main__":
    import asyncio
    import sys

    print(f"API base: {API_BASE}")
    print(f"WebSocket URL: {WS_URL}")

    if ws_connect is None:
        print("Error: websockets package not installed. Try: pip install websockets", file=sys.stderr)
        sys.exit(2)
    if not HAVE_LIBROSA:
        print("Error: librosa not installed. Install test extras or librosa. E.g.: pip install .[test]", file=sys.stderr)
        sys.exit(2)
    if not _server_is_up():
        print("Error: API server not running at /health on localhost:8000. Start it first.", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(test_ws_inference_end_to_end())
        asyncio.run(test_ws_reset_and_reuse())
        print("All checks passed.")
    except Exception as e:
        print(f"Test run failed: {e}", file=sys.stderr)
        raise

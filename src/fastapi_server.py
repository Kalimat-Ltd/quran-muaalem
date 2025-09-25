from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import asdict, is_dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from quran_muaalem import Muaalem, MuaalemOutput
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

# WebSocket protocol
# 1) Client connects to ws://<host>:<port>/ws
# 2) Client sends a JSON "config" message first, e.g.:
#    {
#      "type": "config",
#      "surah": <int>,
#      "ayah": <int>,
#      "start_word": <int>,
#      "end_word" | "num_words": <int>,
#      "rewaya": "hafs",
#      ...madd settings...,
#      "sampling_rate": 16000
#    }
# 3) Client then streams audio as binary frames only: PCM16LE mono at 16kHz.
#    The server accumulates samples until a 2-second chunk (32000 samples) is reached.
# 4) After each new 2s chunk, a rolling window of up to 5 chunks (10s) is built and
#    inference is run. The server replies with a JSON message:
#    {
#       "type":"inference",
#       "final": <bool>,
#       "window_chunks": <int>,
#       "total_samples": <int>,
#       "phonetizer_out": { ... },
#       "result": { ... }
#    }
# 5) Control messages: {"type":"end"}, {"type":"reset"}, {"type":"ping"}

CHUNK_SECS = 2
DEFAULT_SR = 16000
MAX_CHUNKS = 5
AUDIO_FORMAT = "pcm16le"  # fixed wire format for binary frames


def _to_serializable(obj: Any) -> Any:
    """Convert dataclasses, tensors/ndarrays and common containers into JSON-serializable types."""
    # Dataclasses
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}

    # Pydantic v2 / v1
    if hasattr(obj, "model_dump"):
        try:
            return _to_serializable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _to_serializable(obj.dict())
        except Exception:
            pass

    # Torch tensor
    try:
        import torch as _torch  # local alias to avoid masking outer import
        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # Numpy arrays and scalars
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import numpy as _np
        if isinstance(obj, (_np.generic,)):
            return obj.item()
    except Exception:
        pass

    # Builtins & simple types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Common containers
    if isinstance(obj, (list, tuple, set, deque)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    # Enums
    try:
        import enum
        if isinstance(obj, enum.Enum):
            return obj.value
    except Exception:
        pass

    # Generic objects: fall back to __dict__ if available
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_serializable(v) for k, v in vars(obj).items() if not callable(v) and not k.startswith("_")}
        except Exception:
            pass

    # Last resort: string representation
    return repr(obj)


class RollingBuffer:
    def __init__(self, sampling_rate: int, chunk_secs: int = CHUNK_SECS, max_chunks: int = MAX_CHUNKS):
        self.sr = sampling_rate
        self.chunk_size = sampling_rate * chunk_secs
        self.max_chunks = max_chunks
        self._chunks: Deque[np.ndarray] = deque(maxlen=max_chunks)
        self._staging: List[float] = []  # accumulate until reaching chunk_size

    def reset(self) -> None:
        self._chunks.clear()
        self._staging.clear()

    def push_samples(self, samples: np.ndarray) -> List[np.ndarray]:
        """Push new samples (float32 mono in [-1,1]).
        Returns a list of newly formed 2s chunks (0..N) created from staging.
        """
        if samples.ndim > 1:
            samples = samples.squeeze()
        self._staging.extend(samples.astype(np.float32).tolist())
        new_chunks: List[np.ndarray] = []
        while len(self._staging) >= self.chunk_size:
            chunk = np.array(self._staging[: self.chunk_size], dtype=np.float32)
            del self._staging[: self.chunk_size]
            self._chunks.append(chunk)
            new_chunks.append(chunk)
        return new_chunks

    def current_window(self) -> Optional[np.ndarray]:
        if not self._chunks:
            return None
        return np.concatenate(list(self._chunks), axis=0)

    def window_chunk_count(self) -> int:
        return len(self._chunks)

    def total_samples(self) -> int:
        return sum(len(c) for c in self._chunks)

    # New helpers to handle partial data on stream end
    def staging_array(self) -> np.ndarray:
        return np.array(self._staging, dtype=np.float32) if self._staging else np.array([], dtype=np.float32)

    def any_audio_present(self) -> bool:
        return bool(self._chunks) or bool(self._staging)

    def window_with_staging(self) -> Optional[np.ndarray]:
        if not self.any_audio_present():
            return None
        win = self.current_window()
        stag = self.staging_array()
        if win is None:
            return stag
        if stag.size == 0:
            return win
        return np.concatenate([win, stag], axis=0)

    def total_samples_including_staging(self) -> int:
        return self.total_samples() + len(self._staging)


class SessionState:
    def __init__(self, muaalem: Muaalem):
        self.muaalem = muaalem
        self.sr = DEFAULT_SR
        self.aya_ref_text: Optional[str] = None
        self.phonetizer_out = None
        self.buffer = RollingBuffer(self.sr)
        self.lock = asyncio.Lock()

    def configure(self, cfg: Dict[str, Any]) -> None:
        requested_sr = int(cfg.get("sampling_rate", DEFAULT_SR))
        if requested_sr != DEFAULT_SR:
            raise ValueError(f"sampling_rate must be {DEFAULT_SR}")
        self.sr = DEFAULT_SR
        self.buffer = RollingBuffer(self.sr)

        surah = int(cfg["surah"])  # required
        ayah = int(cfg["ayah"])    # required
        start_word = int(cfg.get("start_word", 1))
        end_word = cfg.get("end_word")
        num_words = cfg.get("num_words")

        # Build mushaf attributes (defaults align with test example)
        moshaf = MoshafAttributes(
            rewaya=cfg.get("rewaya", "hafs"),
            madd_monfasel_len=int(cfg.get("madd_monfasel_len", 2)),
            madd_mottasel_len=int(cfg.get("madd_mottasel_len", 4)),
            madd_mottasel_waqf=int(cfg.get("madd_mottasel_waqf", 4)),
            madd_aared_len=int(cfg.get("madd_aared_len", 2)),
        )

        aya_obj = Aya(surah, ayah)
        # Robust: support either an (start, end) API or (start, num_words)
        try:
            if end_word is not None:
                count = int(end_word) - int(start_word) + 1
                if count < 1:
                    count = 1
                uthmani_ref = aya_obj.get_by_imlaey_words(int(start_word), int(count)).uthmani
            elif num_words is not None:
                uthmani_ref = aya_obj.get_by_imlaey_words(int(start_word), int(num_words)).uthmani
            else:
                uthmani_ref = aya_obj.get_by_imlaey_words(int(start_word), 1).uthmani
        except Exception:
            # Fallback to whole ayah if slicing fails
            uthmani_ref = aya_obj.uthmani

        self.aya_ref_text = uthmani_ref
        self.phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

    def decode_binary_audio(self, data: bytes) -> np.ndarray:
        # Expect mono PCM16LE: int16 little-endian -> float32 [-1,1]
        arr = np.frombuffer(data, dtype='<i2')
        return (arr.astype(np.float32) / 32768.0)

def _select_device() -> str:
    """Device priority: cuda > mps (Apple Silicon) > cpu."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _supports_bf16(device: str) -> bool:
    """Return True if bfloat16 is usable on the given device."""
    try:
        if device == "cuda":
            fn = getattr(torch.cuda, "is_bf16_supported", None)
            return bool(fn() if callable(fn) else False)
        if device == "mps":
            # bfloat16 not broadly supported on MPS; prefer float32
            return False
        if device == "cpu":
            # Torch supports bf16 tensors on CPU in many builds; try a tiny op
            try:
                x = torch.ones(1, dtype=torch.bfloat16)
                _ = x + x
                return True
            except Exception:
                return False
    except Exception:
        return False
    return False


def _select_dtype(device: str):
    """Dtype priority: bf16 (if supported) else float32."""
    return torch.bfloat16 if _supports_bf16(device) else torch.float32


app = FastAPI(title="Quran Muaalem WebSocket API")


@app.on_event("startup")
async def _startup() -> None:
    # Load the model once with preferred device/dtype
    device = _select_device()
    dtype = _select_dtype(device)
    app.state.muaalem = Muaalem(device=device, dtype=dtype)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    muaalem: Muaalem = app.state.muaalem
    session = SessionState(muaalem)

    try:
        # Expect initial config
        initial = await ws.receive_text()
        cfg = json.loads(initial)
        if not isinstance(cfg, dict) or cfg.get("type") != "config":
            await ws.send_text(json.dumps({"type": "error", "message": "First message must be a 'config' JSON"}))
            await ws.close(code=1002)
            return
        session.configure(cfg)
        await ws.send_text(json.dumps({"type": "ready", "sampling_rate": session.sr, "audio_format": AUDIO_FORMAT}))

        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                samples = session.decode_binary_audio(msg["bytes"])  # np.float32
                new_chunks = session.buffer.push_samples(samples)
            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"]) if msg["text"] else {}
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                mtype = payload.get("type")
                if mtype == "reset":
                    session.buffer.reset()
                    await ws.send_text(json.dumps({"type": "reset_ack"}))
                    continue
                elif mtype == "end":
                    # Flush any partial audio (including <2s remainder) before closing
                    try:
                        if session.phonetizer_out is not None and session.buffer.any_audio_present():
                            final_wave = session.buffer.window_with_staging()
                            if final_wave is not None and final_wave.size > 0:
                                async with session.lock:
                                    try:
                                        outs: List[MuaalemOutput] = session.muaalem(
                                            [final_wave],
                                            [session.phonetizer_out],
                                            sampling_rate=session.sr,
                                        )
                                    except Exception as e:
                                        await ws.send_text(json.dumps({"type": "error", "message": f"inference_failed: {e}"}))
                                        # proceed to close
                                        outs = []
                                if outs:
                                    out = outs[0]
                                    result: Dict[str, Any] = {
                                        "phonemes": {
                                            "text": getattr(out.phonemes, "text", None),
                                            "probs": _to_serializable(getattr(out.phonemes, "probs", None)),
                                            "ids": _to_serializable(getattr(out.phonemes, "ids", None)),
                                        },
                                        "sifat": [_to_serializable(s) for s in getattr(out, "sifat", [])],
                                    }
                                    await ws.send_text(
                                        json.dumps(
                                            {
                                                "type": "inference",
                                                "final": True,
                                                "phonetizer_out": _to_serializable(session.phonetizer_out),
                                                "window_chunks": session.buffer.window_chunk_count(),
                                                "total_samples": session.buffer.total_samples_including_staging(),
                                                "result": result,
                                            },
                                            ensure_ascii=False,
                                        )
                                    )
                    finally:
                        await ws.send_text(json.dumps({"type": "bye"}))
                        await ws.close()
                        return
                elif mtype == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                    continue
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "Unsupported message type"}))
                    continue
            else:
                # Unknown frame
                await ws.send_text(json.dumps({"type": "error", "message": "Unsupported frame"}))
                continue

            # If we formed one or more 2s chunks, run inference on the current window
            if new_chunks:
                window = session.buffer.current_window()
                if window is None or session.phonetizer_out is None:
                    continue

                
                async with session.lock:
                    try:
                        outs: List[MuaalemOutput] = session.muaalem(
                            [window],
                            [session.phonetizer_out],
                            sampling_rate=session.sr,
                        )
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "error", "message": f"inference_failed: {e}"}))
                        continue

                if not outs:
                    await ws.send_text(json.dumps({"type": "error", "message": "empty_output"}))
                    continue

                out = outs[0]
                result: Dict[str, Any] = {
                    "phonemes": {
                        "text": getattr(out.phonemes, "text", None),
                        "probs": _to_serializable(getattr(out.phonemes, "probs", None)),
                        "ids": _to_serializable(getattr(out.phonemes, "ids", None)),
                    },
                    "sifat": [_to_serializable(s) for s in getattr(out, "sifat", [])],
                }

                await ws.send_text(
                    json.dumps(
                        {
                            "type": "inference",
                            "final": False,
                            "phonetizer_out": _to_serializable(session.phonetizer_out),
                            "window_chunks": session.buffer.window_chunk_count(),
                            "total_samples": session.buffer.total_samples(),
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                )

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
        try:
            await ws.close(code=1011)
        except Exception:
            pass


def main() -> None:
    import uvicorn
    import copy
    try:
        from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG
    except Exception:
        UVICORN_LOGGING_CONFIG = None

    # Start from Uvicorn's default logging config and only add timestamps
    if UVICORN_LOGGING_CONFIG is not None:
        log_config = copy.deepcopy(UVICORN_LOGGING_CONFIG)
        # Uvicorn's formatters use 'fmt' key with its custom formatter classes
        if "formatters" in log_config:
            if "default" in log_config["formatters"]:
                log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s %(message)s"
            if "access" in log_config["formatters"]:
                log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    else:
        # Fallback: minimal config with timestamps, still avoiding duplicates
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"handlers": ["default"], "level": "INFO"},
        }

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        log_config=log_config,
    )


if __name__ == "__main__":
    main()

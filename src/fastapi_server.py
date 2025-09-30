from __future__ import annotations

import asyncio
import json
from collections import deque
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from quran_muaalem import Muaalem, MuaalemOutput
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer
import diff_match_patch as dmp

logger = logging.getLogger(__name__)

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
    MIN_WINDOW_WORDS = 11
    REMAINING_THRESHOLD = 10
    SHIFT_SIZE = 10
    WORD_MATCH_RATIO = 0.7
    FORCE_SLIDE_AFTER = 10
    PREFETCH_THRESHOLD_WORDS = 10
    SLIDE_MARGIN_WORDS = 1

    def __init__(self, muaalem: Muaalem):
        self.muaalem = muaalem
        self.sr = DEFAULT_SR
        self.aya_ref_text: Optional[str] = None
        self.phonetizer_out = None
        self.buffer = RollingBuffer(self.sr)
        self.lock = asyncio.Lock()

        self.moshaf: Optional[MoshafAttributes] = None
        self.aya_obj: Optional[Aya] = None
        self.full_aya_words: List[str] = []
        self.current_window_start_word: int = 1
        self.current_window_word_count: int = 0
        self.window_end_word: int = 0
        self.window_extended_once = False
        self.char_to_word_map: Dict[int, int] = {}
        self._dmp = dmp.diff_match_patch()
        self._post_extension_stall = 0
        self.cumulative_matched_words = 0
        self.global_char_matches: set[int] = set()
        self.full_aya_text: str = ""
        self.full_word_char_offsets: List[int] = []
        self.full_char_to_word_map: Dict[int, int] = {}
        self.full_phonetizer_out = None
        self._full_phoneme_prefix: List[int] = [0]
        self._window_word_char_offsets: List[int] = []
        self._window_global_char_indices: Dict[int, int] = {}
        self._window_global_char_set: set[int] = set()
        self._window_char_matches: set[int] = set()
        self.current_surah: int = 0
        self.current_ayah: int = 0
        self._window_word_global_char_sets: Dict[int, set[int]] = {}
        self._ayah_segments: Deque[Dict[str, Any]] = deque()
        self._next_surah_candidate: int = 0
        self._next_ayah_candidate: int = 0

    @staticmethod
    def _normalize_uthmani_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _build_char_to_word_map(text: str) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        word_idx = -1
        in_word = False
        for idx, ch in enumerate(text):
            if ch.isspace():
                in_word = False
                continue
            if not in_word:
                word_idx += 1
                in_word = True
            mapping[idx] = word_idx
        return mapping

    @staticmethod
    def _normalize_char_map(char_map: List[Any], text_length: int) -> List[Optional[int]]:
        positive_indices = [idx for idx in char_map if isinstance(idx, int) and idx >= 0]
        shift = 0
        if positive_indices:
            if max(positive_indices) >= text_length or (min(positive_indices) >= 1 and 0 not in positive_indices):
                shift = 1

        normalized: List[Optional[int]] = []
        for idx in char_map:
            if not isinstance(idx, int):
                normalized.append(None)
                continue
            adjusted = idx - shift
            if 0 <= adjusted < text_length:
                normalized.append(adjusted)
            else:
                normalized.append(None)
        return normalized

    def _prepare_full_reference(self) -> None:
        self.full_char_to_word_map = {}
        self.full_phonetizer_out = None
        self._full_phoneme_prefix = [0]

        if not self.full_aya_words or self.moshaf is None:
            return

        full_text = self._normalize_uthmani_text(" ".join(self.full_aya_words))
        self.full_aya_text = full_text
        self.full_char_to_word_map = self._build_char_to_word_map(full_text)

        try:
            self.full_phonetizer_out = quran_phonetizer(full_text, self.moshaf, remove_spaces=True)
        except Exception:
            self.full_phonetizer_out = None
            return

        phonemes = getattr(self.full_phonetizer_out, "phonemes", "")
        char_map = getattr(self.full_phonetizer_out, "char_map", [])
        if not isinstance(phonemes, str) or not isinstance(char_map, list):
            return

        normalized_map = self._normalize_char_map(char_map, len(full_text))
        word_count = len(self.full_aya_words)
        if word_count <= 0:
            return

        counts = [0] * word_count
        current_word_idx: Optional[int] = 0
        for map_idx in normalized_map:
            if map_idx is not None:
                mapped_word = self.full_char_to_word_map.get(map_idx)
                if mapped_word is not None:
                    current_word_idx = mapped_word
            if current_word_idx is None or current_word_idx < 0 or current_word_idx >= word_count:
                continue
            counts[current_word_idx] += 1

        prefix: List[int] = [0]
        total = 0
        for count in counts:
            total += count
            prefix.append(total)

        # If some phoneme characters had no mapping, distribute them to the last word
        phoneme_length = len(phonemes)
        if prefix[-1] < phoneme_length and word_count > 0:
            missing = phoneme_length - prefix[-1]
            prefix[-1] += missing
            for idx in range(word_count - 1, -1, -1):
                if counts[idx] > 0 or idx == word_count - 1:
                    for j in range(idx + 1, len(prefix)):
                        prefix[j] += missing
                    break

        self._full_phoneme_prefix = prefix

    def _recompute_full_offsets(self) -> None:
        if not self.full_aya_words:
            self.full_aya_text = ""
            self.full_word_char_offsets = []
            return

        self.full_aya_text = " ".join(self.full_aya_words)
        self.full_word_char_offsets = []
        offset = 0
        for idx, word in enumerate(self.full_aya_words):
            self.full_word_char_offsets.append(offset)
            offset += len(word)
            if idx < len(self.full_aya_words) - 1:
                offset += 1

    def _set_reference_window(self, start_word: int, count: int) -> None:
        if self.aya_obj is None or self.moshaf is None:
            raise RuntimeError("Session not configured")

        total_words = len(self.full_aya_words)
        if total_words == 0:
            raise ValueError("Selected ayah has no words")

        start_word = max(1, start_word)
        start_idx = min(max(start_word - 1, 0), total_words - 1)
        available = total_words - start_idx
        if available <= 0:
            start_idx = 0
            start_word = 1
            available = total_words

        count = max(1, min(count, available))

        window_words = self.full_aya_words[start_idx : start_idx + count]

        if not window_words:
            window_words = self.full_aya_words[:]
            start_idx = 0
            start_word = 1

        normalized_text = self._normalize_uthmani_text(" ".join(window_words))
        words = normalized_text.split()

        if not words or len(words) != len(window_words):
            words = window_words[:]
            normalized_text = " ".join(words)

        self.aya_ref_text = normalized_text
        self.phonetizer_out = quran_phonetizer(normalized_text, self.moshaf, remove_spaces=True)
        self.char_to_word_map = self._build_char_to_word_map(normalized_text)
        self.current_window_start_word = start_idx + 1
        self.current_window_word_count = len(words)
        if self.current_window_word_count > 0:
            self.window_end_word = self.current_window_start_word + self.current_window_word_count - 1
        else:
            self.window_end_word = self.current_window_start_word - 1

        if self.cumulative_matched_words < self.current_window_start_word - 1:
            self.cumulative_matched_words = self.current_window_start_word - 1

        self._window_word_char_offsets = []
        offset = 0
        for idx, word in enumerate(words):
            self._window_word_char_offsets.append(offset)
            offset += len(word)
            if idx < len(words) - 1:
                offset += 1

        self._window_global_char_indices = {}
        self._window_global_char_set = set()
        self._window_word_global_char_sets = {i: set() for i in range(len(words))}
        self._window_char_matches = set()
        for char_idx, word_idx in self.char_to_word_map.items():
            if word_idx is None or word_idx < 0:
                continue
            if word_idx >= len(self._window_word_char_offsets):
                continue
            global_word_idx = self.current_window_start_word - 1 + word_idx
            if global_word_idx < 0 or global_word_idx >= len(self.full_word_char_offsets):
                continue
            local_offset = char_idx - self._window_word_char_offsets[word_idx]
            if local_offset < 0:
                continue
            global_char_idx = self.full_word_char_offsets[global_word_idx] + local_offset
            self._window_global_char_indices[char_idx] = global_char_idx
            self._window_global_char_set.add(global_char_idx)
            try:
                self._window_word_global_char_sets[word_idx].add(global_char_idx)
            except KeyError:
                pass
        if self.global_char_matches:
            retained = self.global_char_matches & self._window_global_char_set
            if retained:
                self._window_char_matches.update(retained)

    def _reference_phoneme_text(self) -> str:
        if self.phonetizer_out is None:
            return ""
        phonemes = getattr(self.phonetizer_out, "phonemes", "")
        if isinstance(phonemes, str):
            return phonemes
        return getattr(phonemes, "text", "") or ""

    def _word_match_progress(self, predicted_phonemes: str) -> tuple[int, int, float]:
        reference = self._reference_phoneme_text()
        if not reference:
            return 0, self.current_window_word_count, 0.0

        diffs = self._dmp.diff_main(reference, predicted_phonemes or "")
        ops: List[str] = []
        for op, data in diffs:
            if op == self._dmp.DIFF_EQUAL:
                ops.extend(["equal"] * len(data))
            elif op == self._dmp.DIFF_DELETE:
                ops.extend(["delete"] * len(data))

        ref_len = len(reference)
        if len(ops) < ref_len:
            ops.extend(["delete"] * (ref_len - len(ops)))
        elif len(ops) > ref_len:
            ops = ops[:ref_len]

        char_map = getattr(self.phonetizer_out, "char_map", []) if self.phonetizer_out is not None else []
        matched_global_chars: set[int] = set()
        for ref_idx, op in enumerate(ops):
            if op != "equal":
                continue
            try:
                char_idx = char_map[ref_idx]
            except Exception:
                char_idx = None
            if not isinstance(char_idx, int):
                continue
            word_idx = self.char_to_word_map.get(char_idx)
            if word_idx is None:
                continue
            global_char_idx = self._window_global_char_indices.get(char_idx)
            if global_char_idx is not None:
                matched_global_chars.add(global_char_idx)

        total_words = self.current_window_word_count
        if matched_global_chars:
            self.global_char_matches.update(matched_global_chars)
            window_matches = matched_global_chars & self._window_global_char_set
            if window_matches:
                self._window_char_matches.update(window_matches)

        window_char_set = self._window_global_char_set
        if window_char_set:
            matched_char_count = len(self._window_char_matches)
            char_ratio = matched_char_count / len(window_char_set)
        else:
            char_ratio = 0.0

        prefix_matched = 0
        for word_idx in range(total_words):
            word_char_set = self._window_word_global_char_sets.get(word_idx)
            if not word_char_set:
                break
            matched_in_word = len(word_char_set & self._window_char_matches)
            ratio = matched_in_word / len(word_char_set)
            if ratio < self.WORD_MATCH_RATIO:
                break
            prefix_matched += 1

        remaining = max(total_words - prefix_matched, 0)
        return prefix_matched, remaining, char_ratio

    def _update_cumulative_progress(self, matched_words: int) -> None:
        if matched_words <= 0:
            return
        total_words = len(self.full_aya_words)
        if total_words <= 0:
            return
        window_start_idx = max(self.current_window_start_word - 1, 0)
        if window_start_idx > self.cumulative_matched_words:
            return
        candidate = window_start_idx + matched_words
        if candidate <= self.cumulative_matched_words:
            return
        self.cumulative_matched_words = min(total_words, candidate)

    def _slide_window_based_on_cumulative(self, *, force: bool = False) -> None:
        total_words = len(self.full_aya_words)
        if total_words == 0 or self.current_window_word_count <= 0:
            return

        matched_from_start = self.cumulative_matched_words - (self.current_window_start_word - 1)
        if not force and matched_from_start < self.SHIFT_SIZE:
            return

        available_after = self._available_words_after_window()
        if available_after <= 0:
            return

        baseline_start = max(1, self.cumulative_matched_words - self.SHIFT_SIZE + 1)
        max_start_allowed = max(1, total_words - self.current_window_word_count + 1)
        max_shift = min(self.SHIFT_SIZE, available_after)
        preferred_start = self.current_window_start_word + max_shift
        minimum_increment = self.current_window_start_word + 1

        if force:
            candidate_start = max(preferred_start, baseline_start, minimum_increment)
        else:
            candidate_start = max(preferred_start, baseline_start)

        new_start = min(candidate_start, max_start_allowed)

        if new_start <= self.current_window_start_word:
            if max_shift <= 0:
                return
            new_start = min(self.current_window_start_word + max_shift, max_start_allowed)
            if new_start <= self.current_window_start_word:
                return

        if not force and new_start - 1 > self.cumulative_matched_words:
            return

        try:
            self._set_reference_window(new_start, self.current_window_word_count)
        except Exception:
            return
        if force and new_start - 1 > self.cumulative_matched_words:
            total_words = len(self.full_aya_words)
            self.cumulative_matched_words = min(total_words, new_start - 1)
        self._post_extension_stall = 0

    def reset_progress(self) -> None:
        self.buffer.reset()
        self.cumulative_matched_words = max(0, self.current_window_start_word - 1)
        self.window_extended_once = False
        self._post_extension_stall = 0
        self.global_char_matches = set()
        self._window_global_char_indices = {}
        self._window_global_char_set = set()
        self._window_word_global_char_sets = {}
        self._window_char_matches = set()

    def _available_words_after_window(self) -> int:
        total_words = len(self.full_aya_words)
        if self.window_end_word <= 0:
            return 0
        return max(total_words - self.window_end_word, 0)

    def _set_next_candidate_after(self, surah: int, ayah: int) -> None:
        if surah <= 0 or ayah <= 0:
            self._next_surah_candidate = 0
            self._next_ayah_candidate = 0
            return
        self._next_surah_candidate = surah
        self._next_ayah_candidate = ayah + 1

    def _append_next_ayah(self) -> bool:
        next_data = self._find_next_ayah_data()
        if not next_data:
            return False

        start_word = len(self.full_aya_words) + 1
        self.full_aya_words.extend(next_data["words"])
        self._recompute_full_offsets()
        self._prepare_full_reference()
        end_word = len(self.full_aya_words)
        self._ayah_segments.append(
            {
                "surah": next_data["surah"],
                "ayah": next_data["ayah"],
                "start_word": start_word,
                "end_word": end_word,
            }
        )
        print(
            f"sliding-debug | appended ayah surah={next_data['surah']} ayah={next_data['ayah']} words={len(next_data['words'])} total_reference_words={len(self.full_aya_words)}"
        )
        return True

    def _ensure_window_capacity(self, start_word: int, required_words: int) -> None:
        if required_words <= 0:
            return

        safe_start = max(start_word, 1)
        guard = 0
        max_iterations = 256

        def available_from_start() -> int:
            prefix = max(safe_start - 1, 0)
            return max(len(self.full_aya_words) - prefix, 0)

        while available_from_start() < required_words:
            if not self._append_next_ayah():
                break
            guard += 1
            if guard >= max_iterations:
                break

    def _update_completed_segments(self) -> None:
        while self._ayah_segments:
            segment = self._ayah_segments[0]
            if self.cumulative_matched_words < segment["end_word"]:
                break
            completed = self._ayah_segments.popleft()
            print(
                "sliding-debug | completed ayah segment surah=%s ayah=%s words=%s-%s"
                % (
                    completed["surah"],
                    completed["ayah"],
                    completed["start_word"],
                    completed["end_word"],
                )
            )

        if self._ayah_segments:
            head = self._ayah_segments[0]
            self.current_surah = head["surah"]
            self.current_ayah = head["ayah"]

    def _find_next_ayah_data(self) -> Optional[Dict[str, Any]]:
        next_surah = self._next_surah_candidate
        next_ayah = self._next_ayah_candidate
        if next_surah <= 0 or next_surah > 114:
            return None

        while next_surah > 0 and next_surah <= 114:
            self._next_surah_candidate = next_surah
            self._next_ayah_candidate = next_ayah
            if next_ayah <= 0:
                next_ayah = 1
            try:
                candidate = Aya(next_surah, next_ayah)
                full_aya = candidate.get()
                raw_words = full_aya.uthmani_words or []
                words = [" ".join(w.split()) for w in raw_words if isinstance(w, str) and w.strip()]
                if words:
                    self._set_next_candidate_after(next_surah, next_ayah)
                    return {
                        "surah": next_surah,
                        "ayah": next_ayah,
                        "words": words,
                    }
            except Exception:
                pass

            if next_surah >= 114:
                break
            next_surah += 1
            next_ayah = 1

        self._next_surah_candidate = 0
        self._next_ayah_candidate = 0
        return None

    def _maybe_prefetch_next_ayah(self, available_after: int, *, threshold: Optional[int] = None) -> int:
        threshold_value = self.PREFETCH_THRESHOLD_WORDS if threshold is None else threshold
        if available_after > threshold_value:
            return available_after

        appended = self._append_next_ayah()
        if appended:
            return self._available_words_after_window()

        return available_after

    def current_offsets(self) -> Dict[str, int]:
        word_idx = max(self.current_window_start_word - 1, 0)
        total_words = len(self.full_aya_words)
        if total_words <= 0:
            return {
                "uthmani_word_offset": 0,
                "uthmani_char_offset": 0,
                "phoneme_char_offset": 0,
            }

        if word_idx >= total_words:
            word_idx = total_words - 1

        uthmani_char_offset = 0
        if 0 <= word_idx < len(self.full_word_char_offsets):
            uthmani_char_offset = self.full_word_char_offsets[word_idx]

        phoneme_char_offset = 0
        if self._full_phoneme_prefix and word_idx < len(self._full_phoneme_prefix):
            phoneme_char_offset = self._full_phoneme_prefix[word_idx]

        return {
            "uthmani_word_offset": word_idx,
            "uthmani_char_offset": uthmani_char_offset,
            "phoneme_char_offset": phoneme_char_offset,
        }

    def _extend_window(self, available_after: int) -> bool:
        if available_after <= 0 or self.current_window_word_count <= 0:
            return False

        add = min(self.SHIFT_SIZE, available_after)
        if add <= 0:
            return False

        previous_count = self.current_window_word_count
        try:
            self._set_reference_window(self.current_window_start_word, self.current_window_word_count + add)
        except Exception:
            return False

        if self.current_window_word_count > previous_count:
            self.window_extended_once = True
            self._post_extension_stall = 0
            print(
                f"sliding-debug | extended window to {self.current_window_word_count} words (start={self.current_window_start_word}, end={self.window_end_word})"
            )
            return True

        return False

    def _load_ayah(self, surah: int, ayah: int, *, preloaded_words: Optional[List[str]] = None) -> None:
        self.current_surah = surah
        self.current_ayah = ayah
        self.aya_obj = Aya(surah, ayah)

        if preloaded_words is not None:
            words = [" ".join(w.split()) for w in preloaded_words if isinstance(w, str) and w.strip()]
        else:
            try:
                full_aya = self.aya_obj.get()
                words = [" ".join(w.split()) for w in (full_aya.uthmani_words or []) if w.strip()]
            except Exception:
                words = []

        self.full_aya_words = words
        self._recompute_full_offsets()

        self._prepare_full_reference()

        if not self.full_aya_words:
            raise ValueError("Unable to resolve words for selected ayah")

        self.global_char_matches = set()
        self._window_char_matches = set()
        self.char_to_word_map = {}
        self._ayah_segments = deque()
        start_word = 1
        end_word = len(self.full_aya_words)
        if end_word >= start_word:
            self._ayah_segments.append(
                {
                    "surah": surah,
                    "ayah": ayah,
                    "start_word": start_word,
                    "end_word": end_word,
                }
            )
        self._set_next_candidate_after(surah, ayah)

    def configure(self, cfg: Dict[str, Any]) -> None:
        requested_sr = int(cfg.get("sampling_rate", DEFAULT_SR))
        if requested_sr != DEFAULT_SR:
            raise ValueError(f"sampling_rate must be {DEFAULT_SR}")
        self.sr = DEFAULT_SR
        self.buffer = RollingBuffer(self.sr)
        self.window_extended_once = False

        surah = int(cfg["surah"])  # required
        ayah = int(cfg["ayah"])    # required
        start_word = max(int(cfg.get("start_word", 1)), 1)
        raw_end_word = cfg.get("end_word")
        raw_num_words = cfg.get("num_words")
        end_word = int(raw_end_word) if raw_end_word is not None else None
        num_words = int(raw_num_words) if raw_num_words is not None else None

        self.moshaf = MoshafAttributes(
            rewaya=cfg.get("rewaya", "hafs"),
            madd_monfasel_len=int(cfg.get("madd_monfasel_len", 2)),
            madd_mottasel_len=int(cfg.get("madd_mottasel_len", 4)),
            madd_mottasel_waqf=int(cfg.get("madd_mottasel_waqf", 4)),
            madd_aared_len=int(cfg.get("madd_aared_len", 2)),
        )

        self._load_ayah(surah, ayah)

        if end_word is not None:
            target_count = end_word - start_word + 1
        elif num_words is not None:
            target_count = num_words
        else:
            target_count = self.MIN_WINDOW_WORDS

        target_count = max(target_count, 1)
        min_initial_window = max(target_count, self.MIN_WINDOW_WORDS)

        self._ensure_window_capacity(start_word, min_initial_window)

        if self.SHIFT_SIZE > 0:
            extended_window = min_initial_window + self.SHIFT_SIZE
            slide_ready_window = extended_window + max(self.SLIDE_MARGIN_WORDS, 1)

            if extended_window > min_initial_window:
                self._ensure_window_capacity(start_word, extended_window)

            if slide_ready_window > extended_window:
                self._ensure_window_capacity(start_word, slide_ready_window)

        total_words = len(self.full_aya_words)

        available_from_start = total_words - max(start_word - 1, 0)
        if available_from_start <= 0:
            start_word = 1
            available_from_start = total_words

        desired_count = min(max(target_count, self.MIN_WINDOW_WORDS), available_from_start)

        try:
            self._set_reference_window(start_word, desired_count)
        except Exception:
            # Fallback to whole ayah if segment retrieval fails
            self._set_reference_window(1, total_words)

        self.cumulative_matched_words = max(0, self.current_window_start_word - 1)
        self._post_extension_stall = 0

    def maybe_extend_reference(self, predicted_phonemes: str) -> None:
        if self.phonetizer_out is None or not self.aya_ref_text or self.current_window_word_count <= 0:
            return

        matched, _, char_ratio = self._word_match_progress(predicted_phonemes or "")
        self._update_cumulative_progress(matched)
        self._update_completed_segments()

        total_words = len(self.full_aya_words)
        if total_words == 0:
            return

        available_after = self._available_words_after_window()
        available_after = self._maybe_prefetch_next_ayah(available_after)
        total_words = len(self.full_aya_words)
        window_char_set = self._window_global_char_set
        window_total_chars = len(window_char_set)
        window_matched_chars = len(window_char_set & self.global_char_matches) if window_char_set else 0

        print(
            f"sliding-debug | window={self.current_window_start_word}-{self.window_end_word} words={self.current_window_word_count} matched_words={matched} ratio={char_ratio:.3f} chars={window_matched_chars}/{window_total_chars} cumulative={self.cumulative_matched_words} available_after={available_after} extended_once={self.window_extended_once}",
        )

        if char_ratio >= self.WORD_MATCH_RATIO:
            if available_after > 0:
                if not self.window_extended_once and self._extend_window(available_after):
                    return

                print(f"sliding-debug | char_ratio>={self.WORD_MATCH_RATIO:.1f} triggering slide")
                prev_start = self.current_window_start_word
                prev_end = self.window_end_word
                prev_count = self.current_window_word_count
                self._slide_window_based_on_cumulative(force=True)

                if (
                    self.current_window_start_word == prev_start
                    and self.window_end_word == prev_end
                    and available_after > 0
                ):
                    shift = min(self.SHIFT_SIZE, available_after)
                    if shift <= 0:
                        shift = 1
                    max_start_allowed = max(1, len(self.full_aya_words) - prev_count + 1)
                    fallback_start = min(prev_start + shift, max_start_allowed)
                    if fallback_start > prev_start:
                        try:
                            self._set_reference_window(fallback_start, prev_count)
                        except Exception:
                            max_count = len(self.full_aya_words) - fallback_start + 1
                            if max_count > 0:
                                try:
                                    self._set_reference_window(fallback_start, max_count)
                                except Exception:
                                    pass

                self._post_extension_stall = 0
            else:
                if total_words > 0 and self.cumulative_matched_words >= total_words:
                    appended = self._append_next_ayah()
                    if appended:
                        self._post_extension_stall = 0
                        return
                    print("sliding-debug | reference exhausted, awaiting next ayah append")
                else:
                    print(
                        f"sliding-debug | window exhausted, awaiting completion (ratio={char_ratio:.3f})"
                    )
            return

        if available_after <= 0:
            total_words = len(self.full_aya_words)
            if total_words > 0 and self.cumulative_matched_words >= total_words:
                appended = self._append_next_ayah()
                if appended:
                    logger.debug(
                        "sliding-debug | appended ayah on window exhaustion window=%s-%s words=%s",
                        self.current_window_start_word,
                        self.window_end_word,
                        self.current_window_word_count,
                    )
                    self._post_extension_stall = 0
                    return
                self._post_extension_stall = min(self._post_extension_stall + 1, self.FORCE_SLIDE_AFTER)
            else:
                self._post_extension_stall = min(self._post_extension_stall + 1, self.FORCE_SLIDE_AFTER)
                print(
                    f"sliding-debug | terminal window waiting for matches (ratio={char_ratio:.3f})"
                )
            return

        if not self.window_extended_once and self._extend_window(available_after):
            return

        self._post_extension_stall += 1
        print(
            f"sliding-debug | stall={self._post_extension_stall}/{self.FORCE_SLIDE_AFTER} (ratio={char_ratio:.3f}) waiting to force slide"
        )
        if self._post_extension_stall >= self.FORCE_SLIDE_AFTER:
            print("sliding-debug | stall threshold reached, forcing slide")
            self._slide_window_based_on_cumulative(force=True)
            self._post_extension_stall = 0

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
                    session.reset_progress()
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
                                    ph_payload: Dict[str, Any] | None = None
                                    if session.phonetizer_out is not None:
                                        ph_payload = {
                                            "phonemes": {
                                                "text": session._reference_phoneme_text(),
                                            },
                                            "sifat": [
                                                _to_serializable(s)
                                                for s in getattr(session.phonetizer_out, "sifat", [])
                                            ],
                                            "char_map": _to_serializable(
                                                getattr(session.phonetizer_out, "char_map", [])
                                            ),
                                        }

                                    offsets = session.current_offsets()

                                    await ws.send_text(
                                        json.dumps(
                                            {
                                                "type": "inference",
                                                "final": True,
                                                "phonetizer_out": ph_payload,
                                                "window_chunks": session.buffer.window_chunk_count(),
                                                "total_samples": session.buffer.total_samples_including_staging(),
                                                "result": result,
                                                "uthmani": session.aya_ref_text,
                                                "offsets": offsets,
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
                current_phonetizer = session.phonetizer_out
                current_uthmani = session.aya_ref_text
                result: Dict[str, Any] = {
                    "phonemes": {
                        "text": getattr(out.phonemes, "text", None),
                        "probs": _to_serializable(getattr(out.phonemes, "probs", None)),
                        "ids": _to_serializable(getattr(out.phonemes, "ids", None)),
                    },
                    "sifat": [_to_serializable(s) for s in getattr(out, "sifat", [])],
                }

                ph_payload: Dict[str, Any] | None = None
                if current_phonetizer is not None:
                    ph_payload = {
                        "phonemes": {
                            "text": session._reference_phoneme_text(),
                        },
                        "sifat": [
                            _to_serializable(s)
                            for s in getattr(current_phonetizer, "sifat", [])
                        ],
                        "char_map": _to_serializable(
                            getattr(current_phonetizer, "char_map", [])
                        ),
                    }

                offsets = session.current_offsets()

                await ws.send_text(
                    json.dumps(
                        {
                            "type": "inference",
                            "final": False,
                            "phonetizer_out": ph_payload,
                            "window_chunks": session.buffer.window_chunk_count(),
                            "total_samples": session.buffer.total_samples(),
                            "result": result,
                            "uthmani": current_uthmani,
                            "offsets": offsets,
                        },
                        ensure_ascii=False,
                    )
                )

                session.maybe_extend_reference(getattr(out.phonemes, "text", "") or "")

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

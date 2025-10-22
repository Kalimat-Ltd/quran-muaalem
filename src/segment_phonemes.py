"""Align Uthmani text to a phoneme stream and infer word boundaries.

This module provides a small edit-distance based aligner and a set of
conservative post-processing passes that adjust word boundaries to better
reflect Arabic tajweed assimilation (idgham/iqlab/ikhfa).

The code aims to be compact and readable: alignment produces a per-character
mapping which is then used to attach phoneme characters to Uthmani letters.
Post-processing is intentionally conservative to avoid creating longer runs
of a base character than exist in the raw phoneme source.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata as ud
from pathlib import Path
import json
from collections import Counter
from typing import List, Tuple, Union

# Characters that can be ignored when aligning because they do not surface
# in the phoneme stream (e.g. silent alif, sukun, shadda markers).
OPTIONAL_UTHMANI_CHARS: set[str] = {
    "ٱ",  # alif wasla – silent in many contexts
    "ْ",  # sukun
    "ّ",  # shadda – handled by doubled consonants in phonemes
    "ٓ",  # maddah
    "ٔ",  # hamza above
    "ٕ",  # hamza below
    "ـ",  # tatweel
    "۟",  # waqf markers
    "ۢ",
    "ۭ",
}

SUN_LETTERS: set[str] = {
    "ت",
    "ث",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ل",
    "ن",
}

# Pairs of characters that we consider equivalent during alignment. This
# keeps the edit cost low when the two writing systems use different symbols
# for the same sound (e.g. different hamza forms, dagger alif vs alif).
EQUIVALENT_PAIRS: set[Tuple[str, str]] = {
    ("إ", "ء"),
    ("أ", "ء"),
    ("ؤ", "ء"),
    ("ئ", "ء"),
    ("ا", "ء"),  # alif pronounced as hamza at word start
    ("ٰ", "ا"),  # dagger alif rendered as a full alif in the phoneme stream
    ("ى", "ا"),
    ("ى", "ۦ"),
    ("ي", "ۦ"),
    ("و", "ۥ"),
    ("ن", "ں"),
    ("م", "۾"),
    ("۾", "م"),
}

HURUF_MUQATTAAT_PHONEMES: dict[str, str] = {
    "الٓمٓ": "ءَلِفلَااااااممممِۦۦۦۦۦۦم",
    "الٓر": "ءَلِفلَاااااامرَاا",
    "المص": "ءَلِفلَااااااممممِۦۦۦۦۦۦمصَاااااادڇ",
    "الر": "ءَلِفلَاااااامرَاا",
    "المر": "ءَلِفلَااااااممممِۦۦۦۦۦۦمرَاا",
    "كهيعص": "كَاااااافهَاايَااعَيييييںںںصَاااااادڇ",
    "طه": "طَااهَاا",
    "طسم": "طَااسِۦۦۦۦۦۦممممِۦۦۦۦۦۦم",
    "طس": "طَااسِۦۦۦۦۦۦن",
    "يس": "يَااسِۦۦۦۦۦۦن",
    "ص": "صَاااااادڇ",
    "حم": "حَاامِۦۦۦۦۦۦم",
    "عسق": "عَيييييںںںسِۦۦۦۦۦۦںںںقَااااااف",
    "ق": "قَااااااف",
    "ن": "نُۥۥۥۥۥۥن",
}


def _substitution_cost(uth_char: str, phoneme_char: str) -> int:
    """Cost for substituting a Uthmani char for a phoneme char.

    Zero cost is used for exact matches or when the pair is considered
    equivalent (see EQUIVALENT_PAIRS). A non-zero cost encourages the DP
    routine to prefer insertions/deletions in ambiguous regions.
    """
    if uth_char == phoneme_char:
        return 0
    if (uth_char, phoneme_char) in EQUIVALENT_PAIRS:
        return 0
    return 2


def _deletion_cost(uth_char: str) -> int:
    return 0 if uth_char in OPTIONAL_UTHMANI_CHARS else 1


def _insertion_cost(_: str) -> int:
    # Every extra phoneme counts equally; treating them uniformly keeps the
    # dynamic programming routine straightforward.
    return 1


def _pronounced_head(word: str) -> str:
    """Return the first pronounced base letter of the Uthmani word.

    This accounts for optional marks and the `lam` + sun-letter pronunciation
    where the following consonant is effectively the pronounced head.
    """

    idx = 0
    length = len(word)
    while idx < length:
        ch = word[idx]
        if ud.combining(ch):
            idx += 1
            continue
        if ch in OPTIONAL_UTHMANI_CHARS:
            idx += 1
            continue
        if ch == "ل":
            lookahead = idx + 1
            while lookahead < length and ud.combining(word[lookahead]):
                lookahead += 1
            if lookahead < length and word[lookahead] in SUN_LETTERS:
                return word[lookahead]
        return ch
    return ""


def _first_base_letter(word: str) -> str:
    """Return the first non-combining, non-optional character in the word."""

    idx = 0
    length = len(word)
    while idx < length:
        ch = word[idx]
        if ud.combining(ch):
            idx += 1
            continue
        if ch in OPTIONAL_UTHMANI_CHARS:
            idx += 1
            continue
        return ch
    return ""


def _last_base_letter(word: str) -> str:
    idx = len(word) - 1
    while idx >= 0:
        ch = word[idx]
        if ud.combining(ch):
            idx -= 1
            continue
        if ch in OPTIONAL_UTHMANI_CHARS:
            idx -= 1
            continue
        return ch
    return ""


def detect_tajweed_rule(prev_word: str, next_word: str) -> str | None:
    """
    Detects basic tajweed assimilation rules between two words.
    Returns one of {'idgham_ghunnah', 'idgham_no_ghunnah', 'ikhfa', 'iqlab', 'clear', None}.
    """
    # Find the last effective base letter index in prev_word (skip combining and optional chars)
    last_base_idx: int | None = None
    for idx in range(len(prev_word) - 1, -1, -1):
        ch = prev_word[idx]
        if ud.combining(ch):
            continue
        if ch in OPTIONAL_UTHMANI_CHARS:
            continue
        last_base_idx = idx
        break
    if last_base_idx is None:
        return None

    last_base = prev_word[last_base_idx]

    # Check for tanwin or explicit nun-sakin markers by inspecting combining marks
    trailing_combining = "".join(
        ch for ch in prev_word[last_base_idx + 1 :] if ud.combining(ch)
    )
    has_tanwin = any(c in {"ً", "ٍ", "ٌ"} for c in trailing_combining)

    # We consider nun-sakin/tanwin cases when the base letter is ن OR when the
    # base has a tanwin combining mark attached.
    if not (last_base == "ن" or has_tanwin):
        return None

    first = _first_base_letter(next_word)
    if not first:
        return None

    if first in {"ي", "ن", "م", "و"}:
        return "idgham_ghunnah"
    if first in {"ر", "ل"}:
        return "idgham_no_ghunnah"
    if first == "ب":
        return "iqlab"
    if first in {
        "ت",
        "ث",
        "ج",
        "د",
        "ذ",
        "ز",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ف",
        "ق",
        "ك",
    }:
        return "ikhfa"
    return "clear"


def _base_letter_counts(word: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    prev_base: str | None = None
    for ch in word:
        if ch == "ّ":  # shadda - double the previous base
            if prev_base is not None:
                counts[prev_base] += 1
        elif ud.combining(ch):
            continue
        elif ch in OPTIONAL_UTHMANI_CHARS:
            continue
        else:
            counts[ch] += 1
            prev_base = ch
    return counts


def _count_base_letters(chars: List[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ch in chars:
        if not ud.combining(ch):
            counts[ch] += 1
    return counts


def _align(
    uthmani_no_spaces: str, phonemes: str
) -> Tuple[int, List[Tuple[str, int | None, int | None]]]:
    """Align the two strings with a standard edit-distance DP."""

    rows, cols = len(uthmani_no_spaces), len(phonemes)
    dp: List[List[int]] = [[0] * (cols + 1) for _ in range(rows + 1)]
    back: List[List[Tuple[str, int | None, int | None] | None]] = [
        [None] * (cols + 1) for _ in range(rows + 1)
    ]

    for row in range(1, rows + 1):
        dp[row][0] = dp[row - 1][0] + _deletion_cost(uthmani_no_spaces[row - 1])
        back[row][0] = ("del", row - 1, 0)
    for col in range(1, cols + 1):
        dp[0][col] = dp[0][col - 1] + _insertion_cost(phonemes[col - 1])
        back[0][col] = ("ins", 0, col - 1)

    for row in range(1, rows + 1):
        uth_char = uthmani_no_spaces[row - 1]
        for col in range(1, cols + 1):
            phon_char = phonemes[col - 1]
            best_cost = dp[row - 1][col - 1] + _substitution_cost(uth_char, phon_char)
            best_op: Tuple[str, int | None, int | None] = ("sub", row - 1, col - 1)

            delete_cost = dp[row - 1][col] + _deletion_cost(uth_char)
            if delete_cost < best_cost:
                best_cost = delete_cost
                best_op = ("del", row - 1, col)

            insert_cost = dp[row][col - 1] + _insertion_cost(phon_char)
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_op = ("ins", row, col - 1)

            dp[row][col] = best_cost
            back[row][col] = best_op

    operations: List[Tuple[str, int | None, int | None]] = []
    row_idx, col_idx = rows, cols
    while row_idx > 0 or col_idx > 0:
        step = back[row_idx][col_idx]
        assert step is not None
        op, uth_idx, phon_idx = step
        operations.append((op, uth_idx, phon_idx))
        if op == "sub":
            row_idx -= 1
            col_idx -= 1
        elif op == "del":
            row_idx -= 1
        elif op == "ins":
            col_idx -= 1
        else:
            raise ValueError(f"Unexpected operation {op!r}")

    operations.reverse()
    return dp[rows][cols], operations


def _build_word_char_mapping(
    word: str, phoneme_word: str
) -> List[dict[str, Union[str, List[str]]]]:
    if not word:
        return []

    _, operations = _align(word, phoneme_word)
    char_groups: List[List[str]] = [[] for _ in word]

    prev_base_index: List[int | None] = []
    last_base: int | None = None
    for idx, ch in enumerate(word):
        if ud.combining(ch) or ch in OPTIONAL_UTHMANI_CHARS:
            prev_base_index.append(last_base)
        else:
            last_base = idx
            prev_base_index.append(last_base)

    def _matches_uthmani_phoneme(
        uth_char: str, phoneme: str, base_char: str | None = None
    ) -> bool:
        if uth_char == "ّ":
            if base_char is None:
                return False
            return _matches_uthmani_phoneme(base_char, phoneme)
        if uth_char == phoneme:
            return True
        if (uth_char, phoneme) in EQUIVALENT_PAIRS:
            return True
        return False

    def _count_matches_for(target_char: str, values: List[str]) -> int:
        return sum(
            1 for value in values if _matches_uthmani_phoneme(target_char, value)
        )

    def _find_insertion_target(phon_char: str, start: int, last_attached: int) -> int:
        for idx in range(start, len(word)):
            ch = word[idx]
            if ch in OPTIONAL_UTHMANI_CHARS and ch != "ّ":
                continue
            base_idx = prev_base_index[idx]
            base_char = word[base_idx] if base_idx is not None else None
            if _matches_uthmani_phoneme(ch, phon_char, base_char):
                return idx
        for idx in range(min(last_attached, len(word) - 1), -1, -1):
            ch = word[idx]
            if ch in OPTIONAL_UTHMANI_CHARS and ch != "ّ":
                continue
            base_idx = prev_base_index[idx]
            base_char = word[base_idx] if base_idx is not None else None
            if _matches_uthmani_phoneme(ch, phon_char, base_char):
                return idx
        return last_attached if 0 <= last_attached < len(word) else len(word) - 1

    uth_pos = phon_pos = 0
    last_attached = 0

    for op, _, _ in operations:
        if op == "sub":
            if uth_pos < len(word) and phon_pos < len(phoneme_word):
                ch = word[uth_pos]
                base_idx = prev_base_index[uth_pos]
                base_char = word[base_idx] if base_idx is not None else None
                phon_char = phoneme_word[phon_pos]
                if _matches_uthmani_phoneme(ch, phon_char, base_char):
                    char_groups[uth_pos].append(phon_char)
                    last_attached = uth_pos
                else:
                    target = _find_insertion_target(
                        phon_char, uth_pos + 1, last_attached
                    )
                    char_groups[target].append(phon_char)
                    last_attached = target
            uth_pos += 1
            phon_pos += 1
        elif op == "del":
            uth_pos += 1
        elif op == "ins":
            if phon_pos < len(phoneme_word):
                target = _find_insertion_target(
                    phoneme_word[phon_pos], uth_pos, last_attached
                )
                char_groups[target].append(phoneme_word[phon_pos])
                last_attached = target
            phon_pos += 1
        else:
            raise ValueError(f"Unexpected operation {op!r}")

    mapping: List[dict[str, Union[str, List[str]]]] = []
    for idx, ch in enumerate(word):
        mapping.append({"char": ch, "phonemes": char_groups[idx]})

    for idx, entry in enumerate(mapping):
        if entry["char"] != "ّ":
            continue
        base_idx = prev_base_index[idx]
        if base_idx is None:
            continue
        base_char = word[base_idx]
        target_list = entry["phonemes"]
        base_list = mapping[base_idx]["phonemes"]

        while _count_matches_for(base_char, base_list) > 1:
            for pos in range(len(base_list) - 1, -1, -1):
                if _matches_uthmani_phoneme(base_char, base_list[pos]):
                    target_list.insert(0, base_list.pop(pos))
                    break
            else:
                break

        for donor_idx in range(base_idx - 1, -1, -1):
            donor_list = mapping[donor_idx]["phonemes"]
            for pos in range(len(donor_list) - 1, -1, -1):
                if _matches_uthmani_phoneme(base_char, donor_list[pos]):
                    target_list.insert(0, donor_list.pop(pos))

    def _next_effective_index(current_index: int) -> int | None:
        nxt_idx = current_index + 1
        while nxt_idx < len(mapping):
            ch = mapping[nxt_idx]["char"]
            if ch in OPTIONAL_UTHMANI_CHARS and ch != "ّ":
                nxt_idx += 1
                continue
            return nxt_idx
        return None

    tanwin_chars = {"ً", "ٍ", "ٌ"}

    def _remove_trailing_alif(values: List[str]) -> bool:
        remaining = values.count("ا")
        if remaining < 2:
            return False
        for pos in range(len(values) - 1, -1, -1):
            if values[pos] == "ا":
                values.pop(pos)
                return True
        return False

    def _next_non_combining_index(start_index: int | None) -> int | None:
        if start_index is None:
            return None
        nxt_idx = start_index + 1
        while nxt_idx < len(mapping):
            ch = mapping[nxt_idx]["char"]
            if ch in OPTIONAL_UTHMANI_CHARS and ch != "ّ":
                nxt_idx += 1
                continue
            if ud.combining(ch):
                nxt_idx += 1
                continue
            return nxt_idx
        return None

    def _share_long_vowel(current_index: int, next_index: int) -> None:
        current_entry = mapping[current_index]
        next_entry = mapping[next_index]

        if current_entry["char"] != "َ":
            return

        next_char = next_entry["char"]
        if next_char not in {"ا", "ٰ"}:
            return

        current_phonemes = current_entry["phonemes"]
        next_phonemes = next_entry["phonemes"]
        if not isinstance(current_phonemes, list) or not isinstance(
            next_phonemes, list
        ):
            return
        if "ا" in current_phonemes:
            return

        alif_count = next_phonemes.count("ا")
        if alif_count < 2:
            return

        base_idx = prev_base_index[current_index]
        base_char = word[base_idx] if base_idx is not None else None
        if base_char == "ء":
            return

        after_idx = _next_effective_index(next_index)
        after_char = mapping[after_idx]["char"] if after_idx is not None else None

        if next_char == "ا":
            if after_char == "ر":
                following_after = _next_non_combining_index(after_idx)
                if following_after is None:
                    return
        else:  # next_char == "ٰ"
            if after_char != "ت":
                return
            diacritic_idx = after_idx + 1 if after_idx is not None else None
            has_tanwin = False
            while diacritic_idx is not None and diacritic_idx < len(mapping):
                char = mapping[diacritic_idx]["char"]
                if char in OPTIONAL_UTHMANI_CHARS and char != "ّ":
                    diacritic_idx += 1
                    continue
                if ud.combining(char):
                    if char in tanwin_chars:
                        has_tanwin = True
                    diacritic_idx += 1
                    continue
                break
            if has_tanwin:
                return

        if not _remove_trailing_alif(next_phonemes):
            return
        current_phonemes.append("ا")

    for idx in range(len(mapping) - 1):
        current = mapping[idx]
        next_idx = _next_effective_index(idx)
        if next_idx is None:
            continue
        nxt = mapping[next_idx]

        _share_long_vowel(idx, next_idx)

        if current["char"] == "َ" and nxt["char"] == "ة":
            donor = current["phonemes"]
            recipient = nxt["phonemes"]
            moved: List[str] = []
            for pos in range(len(donor) - 1, -1, -1):
                phoneme = donor[pos]
                if phoneme in {"َ", "ِ", "ُ"}:
                    continue
                moved.append(donor.pop(pos))
            if moved:
                recipient[:0] = reversed(moved)

    return mapping


def segment_phonemes(
    phoneme_text: str, uthmani_text: str, *, collect_mapping: bool = False
) -> Union[str, Tuple[str, List[dict[str, object]]]]:
    """Return the phoneme stream with word boundaries inferred from the Uthmani text."""

    phoneme_text = phoneme_text.replace('"', "").replace("\r", "").strip()
    uthmani_text = uthmani_text.replace('"', "").replace("\r", " ").strip()

    words = uthmani_text.split()
    if not words:
        return phoneme_text

    compact_uthmani = "".join(words)
    _, operations = _align(compact_uthmani, phoneme_text)

    char_to_word: List[int] = []
    word_ends: List[int] = []
    cursor = 0
    for index, word in enumerate(words):
        cursor += len(word)
        word_ends.append(cursor)
        char_to_word.extend([index] * len(word))

    word_heads: List[str] = [_pronounced_head(word) for word in words]
    word_bases: List[str] = [_first_base_letter(word) for word in words]
    word_last_bases: List[str] = [_last_base_letter(word) for word in words]
    word_base_counts: List[Counter[str]] = [_base_letter_counts(word) for word in words]

    word_segments: List[List[str]] = [[] for _ in words]
    # Track which phoneme index was attached to which word during initial
    # assignment. This lets post-processing check the original phoneme run
    # lengths so we don't accidentally create longer runs than exist in
    # the raw phoneme stream (which was causing extra meems to appear).
    phoneme_owner: List[int | None] = [None] * len(phoneme_text)

    uth_pos = phon_pos = 0
    total_chars = len(char_to_word)

    def next_word_index_from(position: int) -> int | None:
        if position >= total_chars:
            return None
        current_word = char_to_word[position]
        lookahead = position + 1
        while lookahead < total_chars and char_to_word[lookahead] == current_word:
            lookahead += 1
        if lookahead < total_chars:
            return char_to_word[lookahead]
        return None

    current_word = 0
    for op, _, _ in operations:
        if op == "sub":
            if uth_pos < total_chars:
                word_idx = char_to_word[uth_pos]
                next_word_idx = next_word_index_from(uth_pos)
                if (
                    next_word_idx is not None
                    and phoneme_text[phon_pos] != compact_uthmani[uth_pos]
                ):
                    next_head = word_heads[next_word_idx]
                    next_base = word_bases[next_word_idx]
                    # Only assign to next word if:
                    # 1. Phoneme matches next word's head/base
                    # 2. Current word already has enough of this phoneme
                    current_word_base_count = word_base_counts[word_idx]
                    phoneme_char = phoneme_text[phon_pos]
                    if (next_head and phoneme_char == next_head) or (
                        next_base and phoneme_char == next_base
                    ):
                        # Count how many of this phoneme the current word segment already has
                        current_count = sum(
                            1 for ch in word_segments[word_idx] if ch == phoneme_char
                        )
                        allowed_count = current_word_base_count.get(phoneme_char, 0)
                        # Only move to next word if current word is full
                        if current_count >= allowed_count:
                            word_idx = next_word_idx
            else:
                word_idx = len(words) - 1
            current_word = word_idx
            word_segments[word_idx].append(phoneme_text[phon_pos])
            # mark ownership of this phoneme index
            phoneme_owner[phon_pos] = word_idx
            uth_pos += 1
            phon_pos += 1
        elif op == "del":
            uth_pos += 1
        elif op == "ins":
            if words[current_word] in HURUF_MUQATTAAT_PHONEMES:
                word_idx = current_word
            else:
                if uth_pos < total_chars:
                    word_idx = char_to_word[uth_pos]
                    next_word_idx = next_word_index_from(uth_pos)
                    if next_word_idx is not None:
                        next_head = word_heads[next_word_idx]
                        next_base = word_bases[next_word_idx]
                        current_phoneme = phoneme_text[phon_pos]

                        def chars_equivalent(c1: str, c2: str) -> bool:
                            """Check if two characters are equivalent (bidirectional)."""
                            return (
                                c1 == c2
                                or (c1, c2) in EQUIVALENT_PAIRS
                                or (c2, c1) in EQUIVALENT_PAIRS
                            )

                        head_match = next_head and chars_equivalent(
                            current_phoneme, next_head
                        )
                        base_match = next_base and chars_equivalent(
                            current_phoneme, next_base
                        )
                        if head_match or base_match:
                            current_word_end = word_ends[word_idx]
                            if uth_pos >= current_word_end - 1:
                                word_idx = next_word_idx
                else:
                    word_idx = len(words) - 1
            word_segments[word_idx].append(phoneme_text[phon_pos])
            phoneme_owner[phon_pos] = word_idx
            phon_pos += 1
        else:
            raise ValueError(f"Unexpected operation {op!r}")

    for index in range(len(words) - 1):
        if words[index] in HURUF_MUQATTAAT_PHONEMES:
            continue
        next_base = word_bases[index + 1]
        if not next_base:
            continue
        allowed = word_base_counts[index].get(next_base, 0)
        if allowed == 0:
            continue
        segment_chars = word_segments[index]
        if not segment_chars:
            continue
        segment_counts = _count_base_letters(segment_chars)
        current = segment_counts.get(next_base, 0)
        while current > allowed and segment_chars:
            cluster: List[str] = []
            while segment_chars and ud.combining(segment_chars[-1]):
                cluster.insert(0, segment_chars.pop())
            if not segment_chars or segment_chars[-1] != next_base:
                # Put any diacritics back if we cannot pop the base letter.
                segment_chars.extend(cluster)
                break
            cluster.insert(0, segment_chars.pop())
            word_segments[index + 1] = cluster + word_segments[index + 1]
            current -= 1

    assimilation_targets = {"ي"}
    for index in range(len(words) - 1):
        if not word_segments[index + 1]:
            continue
        if words[index] != "أَن":
            continue
        last_base = word_last_bases[index]
        next_base = word_bases[index + 1]
        if last_base != "ن" or not next_base or next_base not in assimilation_targets:
            continue
        cluster: List[str] = []
        chars_next = word_segments[index + 1]
        base_found = False
        while chars_next:
            ch = chars_next.pop(0)
            cluster.append(ch)
            if not ud.combining(ch):
                base_found = True
                break
        while chars_next and ud.combining(chars_next[0]):
            cluster.append(chars_next.pop(0))
        if base_found:
            word_segments[index].extend(cluster)
        else:
            # Put characters back if we did not capture a base letter.
            word_segments[index + 1] = cluster + word_segments[index + 1]

    if phon_pos < len(phoneme_text):
        # append remaining phonemes individually and mark ownership
        for p in range(phon_pos, len(phoneme_text)):
            ch = phoneme_text[p]
            word_segments[-1].append(ch)
            phoneme_owner[p] = len(words) - 1

    # Normalize segments so each list entry is a single character. Some
    # code paths (remaining-tail append) may have added multi-character
    # substrings as single list elements which breaks subsequent per-char
    # pop/insert logic and can lead to duplicated characters. Split any
    # multi-character entries into individual characters here.
    for i, _ in enumerate(word_segments):
        parts = word_segments[i]
        if not parts:
            continue
        new_parts: list[str] = []
        for part in parts:
            if len(part) <= 1:
                new_parts.append(part)
            else:
                # break the multi-char chunk into single-character entries
                new_parts.extend(list(part))
        word_segments[i] = new_parts

    segments = ["".join(parts) for parts in word_segments]

    # Post-process segments to handle tajweed assimilation (idgham/iqlab/ikhfa)
    # Move assimilated phonemes from the end of a previous segment to the start
    # of the next segment when detect_tajweed_rule indicates assimilation.
    def _extract_trailing_suffix_for_base(chars: List[str], base: str) -> List[str]:
        """Return a suffix consisting of one or more base letters equal to `base`
        plus any combining marks that immediately follow those bases. Do NOT
        include combining marks that precede the base (these belong to the
        previous base and should remain).
        The returned list is ordered as it appears (not reversed)."""
        if not chars:
            return []
        suffix: List[str] = []
        i = len(chars) - 1
        # Consume trailing combining marks that belong to a potential moved base
        trailing_combining: List[str] = []
        while i >= 0 and ud.combining(chars[i]):
            trailing_combining.insert(0, chars[i])
            i -= 1

        if i < 0:
            return []

        # If the last non-combining char is the base (or equivalent), include it
        if (
            chars[i] == base
            or (chars[i], base) in EQUIVALENT_PAIRS
            or (base, chars[i]) in EQUIVALENT_PAIRS
        ):
            # include this base and the combining marks that followed it
            suffix.insert(0, chars[i])
            suffix.extend(trailing_combining)
            # Now check if there are additional preceding bases equal to base
            j = i - 1
            while j >= 0:
                # skip combining marks that belong to preceding base
                k = j
                while k >= 0 and ud.combining(chars[k]):
                    k -= 1
                if k < 0:
                    break
                if (
                    chars[k] == base
                    or (chars[k], base) in EQUIVALENT_PAIRS
                    or (base, chars[k]) in EQUIVALENT_PAIRS
                ):
                    # include the combining marks between k and j (these follow that base)
                    between = chars[k + 1 : i + 1]  # safe slice
                    # Prepend this base and its following combining marks to suffix
                    suffix = [chars[k]] + between + suffix
                    j = k - 1
                else:
                    break
            return suffix
        return []

    for idx in range(len(words) - 1):
        rule = detect_tajweed_rule(words[idx], words[idx + 1])
        if rule is None:
            continue
        if rule in {"idgham_ghunnah", "idgham_no_ghunnah"}:
            next_base = word_bases[idx + 1]
            if not next_base:
                continue
            prev_seg = word_segments[idx]
            next_seg = word_segments[idx + 1]
            if not prev_seg or not next_seg:
                continue
            # extract trailing suffix in prev_seg consisting of the next_base
            suffix = _extract_trailing_suffix_for_base(prev_seg, next_base)
            if not suffix:
                continue
            # If the Uthmani previous word's last base equals the next base,
            # keep that final base with the previous word (don't move it).
            prev_word_last_base = word_last_bases[idx]
            if prev_word_last_base and prev_word_last_base == next_base:
                continue

            # Don't move the suffix if doing so would remove the last base
            # letter from the previous segment. Count base letters currently
            # present in prev_seg and what remains after removing suffix.
            def count_bases(chars: List[str]) -> int:
                cnt = 0
                for ch in chars:
                    if not ud.combining(ch) and ch not in OPTIONAL_UTHMANI_CHARS:
                        cnt += 1
                return cnt

            prev_base_count = count_bases(prev_seg)
            # determine how many base letters are in the suffix
            suffix_base_count = 0
            for ch in suffix:
                if not ud.combining(ch) and ch not in OPTIONAL_UTHMANI_CHARS:
                    suffix_base_count += 1

            # Cap the number of bases we move so we don't create a longer
            # contiguous run of the base across the boundary than exists in
            # the raw phoneme text. Compute the actual contiguous run of the
            # base in the raw phoneme string near the boundary using
            # phoneme_owner to locate the first phoneme index for the next
            # word and scanning left/right.
            try:
                first_next_idx = min(
                    i for i, o in enumerate(phoneme_owner) if o == idx + 1
                )
            except ValueError:
                first_next_idx = None
            if first_next_idx is not None:
                # scan left from first_next_idx-1 for contiguous base
                left = 0
                j = first_next_idx - 1
                while j >= 0 and phoneme_text[j] == next_base:
                    left += 1
                    j -= 1
                # scan right from first_next_idx for contiguous base
                right = 0
                k = first_next_idx
                while k < len(phoneme_text) and phoneme_text[k] == next_base:
                    right += 1
                    k += 1
                raw_run = left + right
                # how many bases are already at start of next segment
                leading_next = 0
                for ch in next_seg:
                    if (
                        not ud.combining(ch)
                        and ch not in OPTIONAL_UTHMANI_CHARS
                        and (
                            ch == next_base
                            or (ch, next_base) in EQUIVALENT_PAIRS
                            or (next_base, ch) in EQUIVALENT_PAIRS
                        )
                    ):
                        leading_next += 1
                    else:
                        break
                # cap suffix_base_count so (leading_next + moved) <= raw_run
                allowed_move = max(0, raw_run - leading_next)
                if suffix_base_count > allowed_move:
                    # reduce the suffix to move accordingly
                    # remove trailing base-count suffix elements from suffix
                    new_suffix: List[str] = []
                    # walk suffix left-to-right accumulating until we have
                    # allowed_move base letters
                    base_seen = 0
                    for ch in suffix:
                        if (
                            not ud.combining(ch)
                            and ch not in OPTIONAL_UTHMANI_CHARS
                            and (
                                ch == next_base
                                or (ch, next_base) in EQUIVALENT_PAIRS
                                or (next_base, ch) in EQUIVALENT_PAIRS
                            )
                        ):
                            if base_seen < allowed_move:
                                new_suffix.append(ch)
                                base_seen += 1
                            else:
                                # we stop including bases beyond allowed_move
                                # these should remain in prev_seg
                                break
                        else:
                            # include combining marks that follow included bases
                            if base_seen > 0:
                                new_suffix.append(ch)
                    # if we couldn't include any base, skip moving
                    if not new_suffix:
                        continue
                    suffix = new_suffix
                    # recompute suffix_base_count
                    suffix_base_count = base_seen

            if prev_base_count - suffix_base_count <= 0:
                # moving would leave zero base letters in previous word; skip
                continue

            # remove suffix from prev_seg
            for _ in range(len(suffix)):
                if word_segments[idx]:
                    word_segments[idx].pop()
            # add to front of next_seg
            word_segments[idx + 1] = suffix + word_segments[idx + 1]

    # Rebuild segments after tajweed adjustments
    segments = ["".join(parts) for parts in word_segments]

    # Reverse-fix pass: sometimes alignment placed the base cluster at the
    # start of the next segment instead of keeping it with the previous Uthmani
    # word. If the tajweed rule indicates idgham and the previous word's last
    # base equals the next base, move the leading cluster from next_seg back
    # to prev_seg when prev_seg lacks that base.
    def _extract_leading_cluster_for_base(chars: List[str], base: str) -> List[str]:
        if not chars:
            return []
        cluster: List[str] = []
        # The first non-combining should be the base
        i = 0
        # skip any leading optional chars (unlikely) or combining marks
        while i < len(chars) and (
            ud.combining(chars[i]) or chars[i] in OPTIONAL_UTHMANI_CHARS
        ):
            # keep these in front of next_seg (do not treat as cluster)
            i += 1
        if i >= len(chars):
            return []
        if (
            chars[i] == base
            or (chars[i], base) in EQUIVALENT_PAIRS
            or (base, chars[i]) in EQUIVALENT_PAIRS
        ):
            # collect this base and any combining marks that follow
            cluster.append(chars.pop(i))
            while i < len(chars) and ud.combining(chars[i]):
                cluster.append(chars.pop(i))
            return cluster
        return []

    for idx in range(len(words) - 1):
        rule = detect_tajweed_rule(words[idx], words[idx + 1])
        if rule is None:
            continue
        if rule in {"idgham_ghunnah", "idgham_no_ghunnah"}:
            prev_last_base = word_last_bases[idx]
            next_base = word_bases[idx + 1]
            if not prev_last_base or not next_base:
                continue
            if prev_last_base != next_base:
                continue
            prev_seg = word_segments[idx]
            next_seg = word_segments[idx + 1]

            # if previous segment already contains the base we don't need to do anything
            def contains_base(chars: List[str], base: str) -> bool:
                for ch in chars:
                    if not ud.combining(ch) and ch not in OPTIONAL_UTHMANI_CHARS:
                        if (
                            ch == base
                            or (ch, base) in EQUIVALENT_PAIRS
                            or (base, ch) in EQUIVALENT_PAIRS
                        ):
                            return True
                return False

            if contains_base(prev_seg, prev_last_base):
                continue
            # try to extract leading cluster from next_seg
            cluster = _extract_leading_cluster_for_base(next_seg, prev_last_base)
            if not cluster:
                continue
            # move cluster to end of prev_seg (preserve diacritics that were before)
            word_segments[idx].extend(cluster)

    # Rebuild segments after reverse-fix adjustments
    segments = ["".join(parts) for parts in word_segments]

    # Handle idgham_ghunnah: when prev word ends with nun or tanwin and next
    # begins with a sun/moon base in the idgham set (ي, ن, م, و), the sound is
    # assimilated. Represent this by moving one leading base cluster from the
    # next segment to the previous segment so the previous segment shows the
    # assimilated onset.
    idgham_ghunnah_set = {"ي", "ن", "م", "و"}
    for idx in range(len(words) - 1):
        rule = detect_tajweed_rule(words[idx], words[idx + 1])
        if rule != "idgham_ghunnah":
            continue
        prev_seg = word_segments[idx]
        next_seg = word_segments[idx + 1]
        if not next_seg:
            continue

        # If the next Uthmani word begins with a waw (conjunction 'و'),
        # do not move that leading waw into the previous segment. In many
        # orthographic cases the conjunction should remain attached to the
        # following word for readability even though assimilation may occur
        # in pronunciation.
        if word_bases[idx + 1] == "و":
            continue

        # helper to get first base char of a segment
        def first_base(chars: List[str]) -> str | None:
            for ch in chars:
                if not ud.combining(ch) and ch not in OPTIONAL_UTHMANI_CHARS:
                    return ch
            return None

        next_first = first_base(next_seg)
        if not next_first or next_first not in idgham_ghunnah_set:
            continue

        # Only move the leading cluster when the previous segment lacks a
        # pronounced base (or ends with a hamza). This makes the change
        # conservative: we won't steal a base that the previous Uthmani word
        # is clearly using (fixes cases like 'رَجُلٍ منهم').
        def last_base_in_segment(chars: List[str]) -> str | None:
            for ch in reversed(chars):
                if not ud.combining(ch) and ch not in OPTIONAL_UTHMANI_CHARS:
                    return ch
            return None

        last_prev = last_base_in_segment(prev_seg)
        if last_prev is not None and last_prev != "ء":
            # previous segment has a base other than hamza — do not take
            # the leading base from the next segment.
            continue

        # extract a single leading cluster (base + following combining marks)
        cluster: List[str] = []
        i = 0
        # skip optional/combining at front
        while i < len(next_seg) and (
            ud.combining(next_seg[i]) or next_seg[i] in OPTIONAL_UTHMANI_CHARS
        ):
            i += 1
        if i < len(next_seg) and (
            next_seg[i] == next_first
            or (next_seg[i], next_first) in EQUIVALENT_PAIRS
            or (next_first, next_seg[i]) in EQUIVALENT_PAIRS
        ):
            # pop at index i repeatedly (base then its combining marks)
            cluster.append(next_seg.pop(i))
            while i < len(next_seg) and ud.combining(next_seg[i]):
                cluster.append(next_seg.pop(i))
            # append to prev segment
            word_segments[idx].extend(cluster)

    # Final rebuild of segments after tajweed moves
    segments = ["".join(parts) for parts in word_segments]

    # Final targeted pass: if prev segment has no base (or ends with hamza only)
    # but tajweed indicates idgham_ghunnah, attach a single leading base
    # cluster from the next segment (commonly a ي) to the prev segment so
    # pronunciation matches (e.g. 'ءَ' + 'ي' -> 'ءَي').
    for idx in range(len(words) - 1):
        if detect_tajweed_rule(words[idx], words[idx + 1]) != "idgham_ghunnah":
            continue
        prev_seg = word_segments[idx]
        next_seg = word_segments[idx + 1]
        if not next_seg:
            continue

        # last non-combining char in prev segment
        def last_non_combining(chars: List[str]) -> str | None:
            for ch in reversed(chars):
                if not ud.combining(ch) and ch not in OPTIONAL_UTHMANI_CHARS:
                    return ch
            return None

        last_prev = last_non_combining(prev_seg)
        # only proceed if prev has no base or ends with hamza
        if last_prev is not None and last_prev != "ء":
            continue

        # find leading base in next_seg
        i = 0
        while i < len(next_seg) and (
            ud.combining(next_seg[i]) or next_seg[i] in OPTIONAL_UTHMANI_CHARS
        ):
            i += 1
        if i >= len(next_seg):
            continue
        lead = next_seg[i]
        # Do not pull a leading waw that serves as the conjunction from the
        # next segment into the previous one.
        if word_bases[idx + 1] == "و":
            continue
        if lead not in {"ي", "ن", "م", "و"}:
            continue
        # pop lead and following combining marks
        cluster = [next_seg.pop(i)]
        while i < len(next_seg) and ud.combining(next_seg[i]):
            cluster.append(next_seg.pop(i))
        word_segments[idx].extend(cluster)

    # rebuild segments once more
    segments = ["".join(parts) for parts in word_segments]

    for i, word in enumerate(words):
        if word in HURUF_MUQATTAAT_PHONEMES:
            segments[i] = HURUF_MUQATTAAT_PHONEMES[word]

    segmented_text = " ".join(segments)

    if not collect_mapping:
        return segmented_text

    mapping_output: List[dict[str, object]] = []
    for index, word in enumerate(words):
        phoneme_chunk = segments[index]
        char_mapping = _build_word_char_mapping(word, phoneme_chunk)
        mapping_output.append(
            {
                "word": word,
                "phonemes": phoneme_chunk,
                "chars": char_mapping,
            }
        )

    return segmented_text, mapping_output


def _load_text(value: str) -> str:
    if value.startswith("@"):
        path = value[1:]
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read().strip()
    return value


def _ensure_utf8_io() -> None:
    """Force UTF-8 output so Windows redirection keeps Arabic characters."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7 does not expose reconfigure; rely on defaults.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Insert spaces into a phoneme string by aligning it with an Uthmani Quran text. "
            "Arguments accept raw strings or @path references to UTF-8 files."
        )
    )
    parser.add_argument(
        "--phonemes", "-p", help="Phoneme string or @file", required=False
    )
    parser.add_argument("--uthmani", "-u", help="Uthmani text or @file", required=False)
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="Display the alignment edit cost for diagnostics",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the segmented phoneme text as UTF-8",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write detailed per-word character/phoneme mapping as JSON",
    )
    args = parser.parse_args()

    if args.phonemes and args.uthmani:
        phoneme_text = _load_text(args.phonemes)
        uthmani_text = _load_text(args.uthmani)
        phoneme_lines = [
            line.strip() for line in phoneme_text.split("\n") if line.strip()
        ]
        uthmani_lines = [
            line.strip() for line in uthmani_text.split("\n") if line.strip()
        ]
        if len(phoneme_lines) != len(uthmani_lines):
            print(
                "Error: Number of phoneme lines does not match number of Uthmani lines"
            )
            return
    else:
        # Simple interactive fallback to keep the script convenient during ad-hoc use.
        phoneme_text = input("Enter phoneme text (no spaces): ").strip()
        uthmani_text = input("Enter Uthmani text (with spaces): ").strip()
        phoneme_lines = [phoneme_text]
        uthmani_lines = [uthmani_text]

    _ensure_utf8_io()

    collect_mapping = bool(args.json_output)
    segmented_lines = []
    mappings = []
    for p_line, u_line in zip(phoneme_lines, uthmani_lines):
        result = segment_phonemes(p_line, u_line, collect_mapping=collect_mapping)
        if collect_mapping:
            seg, mapping = result
            segmented_lines.append(seg)
            mappings.extend(mapping)
        else:
            segmented_lines.append(result)

    segmented = "\n".join(segmented_lines)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(segmented, encoding="utf-8")
        print(f"Wrote segmented phoneme text to {output_path}")
    else:
        print(segmented)

    if collect_mapping and args.json_output:
        json_path = Path(args.json_output)
        json_path.write_text(
            json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote mapping JSON to {json_path}")

    if args.show_cost and len(phoneme_lines) == 1:
        compact_uthmani = "".join(uthmani_lines[0].split())
        cost, _ = _align(compact_uthmani, phoneme_lines[0])
        print(f"Alignment cost: {cost}")


if __name__ == "__main__":
    main()

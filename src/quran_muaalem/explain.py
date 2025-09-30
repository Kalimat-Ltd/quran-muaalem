from dataclasses import dataclass, asdict
from typing import Literal
import html
import json
import os
from pathlib import Path

from quran_transcript import SifaOutput
import quran_transcript.alphabet as alph
import diff_match_patch as dmp
from rich import print
from rich.text import Text
from rich.table import Table
from rich.console import Console

from .muaalem_typing import Sifa
from .modeling.vocab import SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS


class _ExplainSessionState:
    """Track cumulative Uthmani text, correctness, and rendered HTML."""

    def __init__(self) -> None:
        self.text: str = ""
        self.correct_indices: set[int] = set()
        self.html_segments: list[str] = []
        self.html_path: Path = Path(os.environ.get("EXPLAIN_HTML_PATH", "build/uthmani_alignments.html"))

    @staticmethod
    def _suffix_prefix_overlap(left: str, right: str) -> int:
        max_len = min(len(left), len(right))
        for length in range(max_len, 0, -1):
            if left.endswith(right[:length]):
                return length
        return 0

    def reset(self) -> None:
        self.text = ""
        self.correct_indices.clear()
        self.html_segments = []
        self.html_path = Path(os.environ.get("EXPLAIN_HTML_PATH", "build/uthmani_alignments.html"))
        try:
            if self.html_path.exists():
                self.html_path.unlink()
        except Exception:
            pass

    def integrate_window(self, window: str) -> int:
        """Ensure the session text contains ``window`` and return its offset."""

        if not window:
            return 0

        if not self.text:
            self.text = window
            return 0

        idx = self.text.find(window)
        if idx != -1:
            return idx

        idx = window.find(self.text)
        if idx != -1:
            shift = idx
            if shift:
                self.correct_indices = {pos + shift for pos in self.correct_indices}
            self.text = window
            return 0

        suffix_overlap = self._suffix_prefix_overlap(self.text, window)
        if suffix_overlap:
            offset = len(self.text) - suffix_overlap
            self.text += window[suffix_overlap:]
            return offset

        prefix_overlap = self._suffix_prefix_overlap(window, self.text)
        if prefix_overlap:
            shift = len(window) - prefix_overlap
            if shift:
                self.correct_indices = {pos + shift for pos in self.correct_indices}
            self.text = window + self.text[prefix_overlap:]
            return 0

        # No meaningful overlap – treat as a new session scope.
        self.reset()
        self.text = window
        return 0

    def append_html(
        self,
        phoneme_segments: list[tuple[str, str]],
        uthmani_segments: list[tuple[str, str]],
    ) -> None:
        snippet = self._render_html_snippet(phoneme_segments, uthmani_segments)
        self.html_segments.append(snippet)
        self._write_html()

    def _render_html_snippet(
        self,
        phoneme_segments: list[tuple[str, str]],
        uthmani_segments: list[tuple[str, str]],
    ) -> str:
        def render_spans(segments: list[tuple[str, str]], base_class: str) -> str:
            spans = []
            for cls, text in segments:
                if not text:
                    continue
                escaped = html.escape(text)
                spans.append(f"<span class='{base_class} {cls}'>{escaped}</span>")
            return "".join(spans)

        phoneme_html = render_spans(phoneme_segments, "segment phoneme")
        uthmani_html = render_spans(uthmani_segments, "segment uthmani")

        return (
            "<section class='inference'>"
            "<div class='inference__block inference__block--phonemes'>"
            "<h3>Phoneme Diff</h3>"
            f"<div class='segment-container'>{phoneme_html or '<em>No phonemes</em>'}</div>"
            "</div>"
            "<div class='inference__block inference__block--uthmani'>"
            "<h3>Uthmani Alignment</h3>"
            f"<div class='segment-container'>{uthmani_html or '<em>No text</em>'}</div>"
            "</div>"
            "</section>"
        )

    def _write_html(self) -> None:
        self.html_path.parent.mkdir(parents=True, exist_ok=True)
        with self.html_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "<!DOCTYPE html>\n"
                "<html lang='en'>\n<head>\n<meta charset='utf-8'/>\n"
                "<title>Uthmani Alignment Explanations</title>\n"
                "<style>\n"
                "body{font-family:'Segoe UI',sans-serif;background:#101418;color:#f1f5f9;margin:0;padding:2rem;}\n"
                "h3{margin:0 0 .5rem;font-size:1rem;color:#d1e7ff;}\n"
                ".inference{background:#1b2430;border-radius:12px;padding:1rem;margin-bottom:1.5rem;box-shadow:0 4px 12px rgba(15,23,42,0.35);}\n"
                ".inference__block{margin-bottom:1rem;}\n"
                ".segment-container{background:#0f172a;border-radius:8px;padding:.75rem;white-space:pre-wrap;word-break:break-word;font-size:1.05rem;}\n"
                ".segment{padding:0 .08rem;}\n"
                ".phoneme.match{color:#38ef7d;}\n"
                ".phoneme.insert{color:#60a5fa;}\n"
                ".phoneme.delete{color:#f87171;text-decoration:line-through;}\n"
                ".uthmani.correct{color:#e2e8f0;}\n"
                ".uthmani.partial{color:#fbbf24;}\n"
                ".uthmani.missing{color:#f87171;text-decoration:line-through;}\n"
                ".uthmani.unreferenced{color:#64748b;}\n"
                "</style>\n</head>\n<body>\n"
            )
            for segment in self.html_segments:
                fh.write(segment)
                fh.write("\n")
            fh.write("</body>\n</html>\n")


_SESSION_STATE = _ExplainSessionState()


def reset_explain_session() -> None:
    """Reset cached explain session state (useful between runs/tests)."""

    _SESSION_STATE.reset()


@dataclass
class PhonemeGroup:
    ref: str = ""
    out: str = ""
    ref_idx: int | None = None
    out_idx: int | None = None
    tag: Literal["exact", "partial", "insert", "delete"] | None = None

    def get_tag(self):
        if self.ref == "" and self.out == "":
            raise ValueError("The Entire group is empty")
        if self.ref == self.out:
            self.tag = "exact"
        elif self.ref != "" and self.out == "":
            self.tag = "delete"
        elif self.out != "" and self.ref == "":
            self.tag = "insert"
        else:
            self.tag = "partial"
        return self.tag


def merge_same_phoneme_group(ph_groups: list[PhonemeGroup]) -> list[PhonemeGroup]:
    outs = [ph_groups[0]]
    prev_idx = 0
    for curr_idx in range(1, len(ph_groups)):
        # out is part of ref
        if (
            ph_groups[prev_idx].out_idx is not None
            and ph_groups[curr_idx].ref_idx is not None
            and ph_groups[prev_idx].out in ph_groups[curr_idx].ref
        ):
            del outs[-1]
            outs.append(
                PhonemeGroup(
                    ref=ph_groups[curr_idx].ref,
                    ref_idx=ph_groups[curr_idx].ref_idx,
                    out=ph_groups[prev_idx].out,
                    out_idx=ph_groups[prev_idx].out_idx,
                )
            )
        # ref is part of out
        elif (
            ph_groups[prev_idx].ref_idx is not None
            and ph_groups[curr_idx].out_idx is not None
            and ph_groups[prev_idx].ref in ph_groups[curr_idx].out
        ):
            del outs[-1]
            outs.append(
                PhonemeGroup(
                    ref=ph_groups[prev_idx].ref,
                    ref_idx=ph_groups[prev_idx].ref_idx,
                    out=ph_groups[curr_idx].out,
                    out_idx=ph_groups[curr_idx].out_idx,
                )
            )
        else:
            outs.append(ph_groups[curr_idx])
        prev_idx = curr_idx
    return outs


def segment_groups(
    ref_groups: list[str],
    groups: list[str],
    diffs,
) -> list[PhonemeGroup]:
    """Join similar phonmes groups and diffrentiate between groups"""
    ref_counter = 0
    ref_ptr = 0
    ref_group_idx = 0
    out_counter = 0
    out_ptr = 0
    out_group_idx = 0

    out_pairs = []
    for op, data in diffs:
        if op == 0:
            ref_counter += len(data)
            out_counter += len(data)
        elif op == 1:
            out_counter += len(data)
        elif op == -1:
            ref_counter += len(data)

        ref_has_match = True
        out_has_match = True
        while ref_has_match or out_has_match:
            pair = PhonemeGroup()
            if ref_group_idx < len(ref_groups):
                if (ref_counter - ref_ptr) >= len(ref_groups[ref_group_idx]):
                    pair.ref = ref_groups[ref_group_idx]
                    pair.ref_idx = ref_group_idx
                    ref_ptr += len(ref_groups[ref_group_idx])
                    ref_group_idx += 1
                else:
                    ref_has_match = False
            else:
                ref_has_match = False

            if out_group_idx < len(groups):
                if (out_counter - out_ptr) >= len(groups[out_group_idx]):
                    pair.out = groups[out_group_idx]
                    pair.out_idx = out_group_idx
                    out_ptr += len(groups[out_group_idx])
                    out_group_idx += 1
                else:
                    out_has_match = False
            else:
                out_has_match = False

            if pair.ref or pair.out:
                out_pairs.append(pair)
    return merge_same_phoneme_group(out_pairs)


def expalin_sifat(
    sifat: list[Sifa],
    exp_sifat: list[SifaOutput],
    diffs,
):
    table = []
    chunks = [s.phonemes_group for s in sifat]
    exp_chunks = [s.phonemes for s in exp_sifat]

    groups = segment_groups(ref_groups=exp_chunks, groups=chunks, diffs=diffs)
    keys = set(asdict(sifat[0]).keys()) - {"phonemes_group"}
    madd_group = alph.phonetics.alif + alph.phonetics.yaa_madd + alph.phonetics.waw_madd

    for group in groups:
        raw = {}
        tag = group.get_tag()
        if (tag == "exact") or (tag == "partial" and group.ref[0] in madd_group):
            raw["tag"] = "exact"
            raw["phonemes"] = sifat[group.out_idx].phonemes_group
            raw["exp_phonemes"] = exp_sifat[group.ref_idx].phonemes
            for key in keys:
                if getattr(sifat[group.out_idx], key) is not None:
                    raw[f"{key}"] = getattr(sifat[group.out_idx], key).text
                else:
                    raw[f"{key}"] = "None"

                raw[f"exp_{key}"] = getattr(exp_sifat[group.ref_idx], key)
        elif tag in {"partial", "insert"}:
            raw["tag"] = "insert"
            raw["phonemes"] = sifat[group.out_idx].phonemes_group
            raw["exp_phonemes"] = ""
            for key in keys:
                if getattr(sifat[group.out_idx], key) is not None:
                    raw[f"{key}"] = getattr(sifat[group.out_idx], key).text
                else:
                    raw[f"{key}"] = "None"

                raw[f"exp_{key}"] = ""
        if raw:
            table.append(raw)

    # print(json.dumps(table, indent=2, ensure_ascii=False))
    return table


def print_sifat_table(
    table: list[dict],
    lang: Literal["arabic", "english"] = "arabic",
):
    """Print the sifat comparison table with rich highlighting"""
    if not table:
        return

    # Create a rich Table
    rich_table = Table()

    # Get base columns (non-exp keys without 'tag')
    base_keys = [k for k in table[0].keys() if not k.startswith("exp_") and k != "tag"]

    # Add columns
    # rich_table.add_column("Tag", style="cyan")
    for key in base_keys:
        rich_table.add_column(key.replace("_", " ").title())

    # Add rows
    for row in table:
        tag = row["tag"]
        values = []
        for key in base_keys:
            exp_key = f"exp_{key}"
            value = str(row[key])
            if key != "phonemes" and lang == "arabic":
                value = SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS[value]

            # Apply styling based on tag and comparison
            if tag == "exact" and row.get(exp_key) != row[key]:
                values.append(f"[red]{value}[/red]")
            elif tag == "insert":
                values.append(f"[yellow]{value}[/yellow]")
            else:
                values.append(value)

        rich_table.add_row(*values)

    # Print the table
    console = Console()
    console.print(rich_table)


def explain_for_terminal(
    phonemes: str,
    exp_phonemes: str,
    sifat: list[Sifa],
    exp_sifat: list[SifaOutput],
    lang: Literal["arabic", "english"] = "english",
):
    # Create diff-match-patch object
    dmp_obj = dmp.diff_match_patch()

    # Calculate differences
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)

    # Create a Rich Text object for colored output
    result = Text()

    # Process each difference
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            result.append(data, style="white")
        elif op == dmp_obj.DIFF_INSERT:
            result.append(data, style="green")
        elif op == dmp_obj.DIFF_DELETE:
            result.append(data, style="red strike")

    # Print the result
    print(result)
    sifat_table = expalin_sifat(sifat, exp_sifat, diffs)
    print_sifat_table(sifat_table, lang=lang)  # Add this line to print the table


def explain_terminal_new(
    phonemes: str,
    exp_phonemes: str,
    exp_char_map: list[int | None],
    uthmani_text: str,
):
    """Explain results in terminal and print a Uthmani-aligned, colorized view.

    Behavior:
    - Computes diffs between the reference phonemes (from the phonetizer) and the
      predicted phonemes.
    - Uses exp_char_map to map each reference phoneme back to an index in the
      original Uthmani text.
    - Styles each Uthmani character according to whether the mapped phonemes are
      exact (white), deleted/missing (red strike), or mixed/partial (yellow).
    - Prints the phoneme-level diff (colored) and the styled Uthmani line, then
      prints the sifat comparison table (reusing existing helpers).

    Args:
        phonemes: predicted phonemes string (model output)
        exp_phonemes: expected/reference phonemes string (phonetizer output)
        exp_char_map: list mapping each char in exp_phonemes -> index in uthmani_text
        uthmani_text: the original Uthmani text string
    """
    # Build diffs (reference = exp_phonemes, predicted = phonemes)
    dmp_obj = dmp.diff_match_patch()
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)

    # 1) Print phoneme-level diff (same as explain_for_terminal but keep it here)
    phoneme_text = Text()
    phoneme_segments: list[tuple[str, str]] = []
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            phoneme_text.append(data, style="white")
            phoneme_segments.append(("match", data))
        elif op == dmp_obj.DIFF_INSERT:
            phoneme_text.append(data, style="green")
            phoneme_segments.append(("insert", data))
        elif op == dmp_obj.DIFF_DELETE:
            phoneme_text.append(data, style="red strike")
            phoneme_segments.append(("delete", data))

    print(phoneme_text)

    window_text = uthmani_text or ""
    window_offset = _SESSION_STATE.integrate_window(window_text)

    # 2) Map reference phoneme ops to indices in exp_phonemes
    # Create an array of operations aligned to each index of exp_phonemes.
    ops_per_ref_idx: list[str] = []
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            ops_per_ref_idx.extend(["equal"] * len(data))
        elif op == dmp_obj.DIFF_DELETE:
            ops_per_ref_idx.extend(["delete"] * len(data))
        elif op == dmp_obj.DIFF_INSERT:
            # Inserts correspond to predicted-only phonemes and do not consume
            # reference indices, so they do not affect ops_per_ref_idx length.
            continue

    # Truncate/pad to reference length
    if len(ops_per_ref_idx) < len(exp_phonemes):
        ops_per_ref_idx.extend(["equal"] * (len(exp_phonemes) - len(ops_per_ref_idx)))
    elif len(ops_per_ref_idx) > len(exp_phonemes):
        ops_per_ref_idx = ops_per_ref_idx[: len(exp_phonemes)]

    # 3) Aggregate per-Uthmani-index styles because multiple reference phonemes
    # might map to the same Uthmani character.
    positive_indices = [
        idx for idx in exp_char_map if isinstance(idx, int) and idx >= 0
    ]
    needs_shift = False
    if window_text and positive_indices:
        max_idx = max(positive_indices)
        min_idx = min(positive_indices)
        if max_idx >= len(window_text):
            needs_shift = True
        elif min_idx >= 1 and 0 not in positive_indices:
            needs_shift = True

    uth_style_counts: dict[int, dict[str, int]] = {}
    newly_correct: set[int] = set()
    for ref_idx, op in enumerate(ops_per_ref_idx):
        try:
            map_idx = exp_char_map[ref_idx] if ref_idx < len(exp_char_map) else None
        except Exception:
            map_idx = None
        if map_idx is None or not isinstance(map_idx, int):
            continue

        uth_idx = map_idx - 1 if needs_shift else map_idx

        if uth_idx < 0 or uth_idx >= len(window_text):
            continue

        global_idx = window_offset + uth_idx
        if uth_idx not in uth_style_counts:
            uth_style_counts[uth_idx] = {"equal": 0, "delete": 0}
        if op == "equal":
            uth_style_counts[uth_idx]["equal"] += 1
            newly_correct.add(global_idx)
        elif op == "delete":
            uth_style_counts[uth_idx]["delete"] += 1

    if newly_correct:
        _SESSION_STATE.correct_indices.update(newly_correct)

    # 4) Produce styled Uthmani text
    uth_text = Text()
    uthmani_segments: list[tuple[str, str]] = []
    for idx, ch in enumerate(window_text):
        global_idx = window_offset + idx
        counts = uth_style_counts.get(idx)
        if global_idx in _SESSION_STATE.correct_indices:
            style = "white"
            html_class = "correct"
        elif counts is None:
            # No mapping -> dim (unreferenced character / diacritics)
            style = "dim"
            html_class = "unreferenced"
        else:
            e = counts.get("equal", 0)
            d = counts.get("delete", 0)
            if e > 0 and d == 0:
                style = "white"
                html_class = "correct"
            elif d > 0 and e == 0:
                style = "red strike"
                html_class = "missing"
            else:
                # Mixed tags on same Uthmani char: partial mismatch
                style = "yellow"
                html_class = "partial"
        uth_text.append(ch, style=style)
        uthmani_segments.append((html_class, ch))

    # Print a labeled Uthmani line
    console = Console()
    console.print("\nUthmani-aligned view:")
    console.print(uth_text)

    _SESSION_STATE.append_html(phoneme_segments, uthmani_segments)

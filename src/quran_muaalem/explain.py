from quran_transcript import SifaOutput
import diff_match_patch as dmp
from rich import print
from rich.text import Text

from .muaalem_typing import Sifa


def explain_for_terminal(
    phonemes: str, exp_phonemes: str, sifat: list[Sifa], exp_sifat: list[SifaOutput]
):
    # Create diff-match-patch object
    dmp_obj = dmp.diff_match_patch()

    # Calculate differences
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)
    dmp_obj.diff_cleanupSemantic(diffs)

    # Create a Rich Text object for colored output
    result = Text()

    # Process each difference
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:  # Use dmp_obj instead of dmp
            result.append(data, style="white")
        elif op == dmp_obj.DIFF_INSERT:  # Use dmp_obj instead of dmp
            result.append(data, style="green")
        elif op == dmp_obj.DIFF_DELETE:  # Use dmp_obj instead of dmp
            result.append(data, style="red strike")

    # Print the result
    print(result)

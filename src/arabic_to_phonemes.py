"""
Convert Arabic text to phonemes without sifat processing.

This script takes Arabic text (Uthmani or regular Arabic) and converts it to
phonetic representation using the Quran phonetizer operations, but skips the
sifat (phonetic attributes) processing which can cause errors with non-Quranic text.

Usage (PowerShell):
    # Basic usage with text argument
    python arabic_to_phonemes.py --text "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

    # Without spaces (concatenated phonemes)
    python src/arabic_to_phonemes.py --text "وَجَنَّـٰتٍ أَلْفَافًاإِنَّ"

    # From a file
    python arabic_to_phonemes.py --input-file input.txt --output-file output.txt

    # Interactive mode
    python arabic_to_phonemes.py --interactive

    # Custom madd lengths (monfasel, mottasel, mottasel_waqf, aared)
    python arabic_to_phonemes.py --text "الْحَمْدُ لِلَّهِ" --madd-config 4 5 6 4

Examples:
    # Single verse
    python arabic_to_phonemes.py --text "قُلْ هُوَ اللَّهُ أَحَدٌ"

    # Multiple verses from file
    python arabic_to_phonemes.py --input-file verses.txt --output-file phonemes.txt --no-spaces
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure src/ is importable
THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quran_transcript import MoshafAttributes, alphabet as alph, quran_phonetizer
from quran_transcript.phonetics.operations import OPERATION_ORDER


def arabic_to_phonemes(
    arabic_text: str,
    moshaf: MoshafAttributes,
    remove_spaces: bool = False
) -> str:
    """
    Convert Arabic text to phonemes without sifat processing.
    
    Args:
        arabic_text: Input Arabic text (Uthmani or regular Arabic)
        moshaf: Moshaf attributes for Tajweed rules
        remove_spaces: If True, remove all spaces from output
    
    Returns:
        Phonetized text string
    """
    text = arabic_text
    
    # Clean extra spaces
    text = re.sub(r"\s+", alph.uthmani.space, text)
    text = re.sub(r"(\s$|^\s)", "", text)
    
    # Apply all phonetic operations
    for op in OPERATION_ORDER:
        text = op.apply(text, moshaf)
    
    # Remove spaces if requested
    if remove_spaces:
        text = text.replace(alph.uthmani.space, "")
    
    return text


def create_moshaf(madd_config: tuple[int, int, int, int] | None = None) -> MoshafAttributes:
    """
    Create MoshafAttributes with specified or default madd lengths.
    
    Args:
        madd_config: Tuple of (monfasel, mottasel, mottasel_waqf, aared) lengths
                    Default is (4, 4, 4, 4)
    
    Returns:
        Configured MoshafAttributes instance
    """
    if madd_config is None:
        madd_config = (4, 4, 4, 4)
    
    return MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=madd_config[0],
        madd_mottasel_len=madd_config[1],
        madd_mottasel_waqf=madd_config[2],
        madd_aared_len=madd_config[3],
    )


def process_text(
    text: str,
    moshaf: MoshafAttributes,
    remove_spaces: bool = False,
    show_original: bool = True
) -> str:
    """
    Process a single line of text and return formatted output.
    
    Args:
        text: Input Arabic text
        moshaf: Moshaf attributes
        remove_spaces: Whether to remove spaces
        show_original: Whether to include original text in output
    
    Returns:
        Formatted output string
    """
    text = text.strip()
    if not text:
        return ""
    
    phonemes = arabic_to_phonemes(text, moshaf, remove_spaces)
    
    if show_original:
        return f"Original: {text}\nPhonemes: {phonemes}\n"
    else:
        return phonemes


def process_file(
    input_path: Path,
    output_path: Path,
    moshaf: MoshafAttributes,
    remove_spaces: bool = False,
    show_original: bool = True
) -> None:
    """
    Process an entire file of Arabic text.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        moshaf: Moshaf attributes
        remove_spaces: Whether to remove spaces
        show_original: Whether to include original text
    """
    with input_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()
    
    results = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            results.append("")
            continue
        
        try:
            result = process_text(line, moshaf, remove_spaces, show_original)
            results.append(result)
        except Exception as e:
            print(f"Error processing line {i}: {e}", file=sys.stderr)
            results.append(f"ERROR: {line}")
    
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.write("\n".join(results))
    
    print(f"Processed {len(lines)} lines")
    print(f"Output written to: {output_path}")


def interactive_mode(moshaf: MoshafAttributes, remove_spaces: bool = False) -> None:
    """
    Run in interactive mode, processing text from user input.
    
    Args:
        moshaf: Moshaf attributes
        remove_spaces: Whether to remove spaces
    """
    print("=" * 70)
    print("Arabic to Phonemes - Interactive Mode")
    print("=" * 70)
    print("Enter Arabic text (or 'quit'/'exit' to stop)")
    print("Press Enter twice to process multiple lines")
    print()
    
    while True:
        try:
            print("Enter text: ", end="")
            text = input().strip()
            
            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            phonemes = arabic_to_phonemes(text, moshaf, remove_spaces)
            print(f"\nOriginal: {text}")
            print(f"Phonemes: {phonemes}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Arabic text to phonemes without sifat processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single text
  python arabic_to_phonemes.py --text "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
  
  # Without spaces
  python arabic_to_phonemes.py --text "الْحَمْدُ لِلَّهِ" --no-spaces
  
  # Process file
  python arabic_to_phonemes.py --input-file verses.txt --output-file phonemes.txt
  
  # Interactive mode
  python arabic_to_phonemes.py --interactive
  
  # Custom madd lengths
  python arabic_to_phonemes.py --text "قُلْ هُوَ اللَّهُ أَحَدٌ" --madd-config 2 4 4 2
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Arabic text to convert to phonemes"
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to input file containing Arabic text (one line per entry)"
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # Output options
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to output file (required when using --input-file)"
    )
    parser.add_argument(
        "--no-spaces",
        action="store_true",
        help="Remove all spaces from phoneme output (concatenated phonemes)"
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Don't include original text in output (only phonemes)"
    )
    
    # Tajweed configuration
    parser.add_argument(
        "--madd-config",
        nargs=4,
        type=int,
        metavar=("MONFASEL", "MOTTASEL", "MOTTASEL_WAQF", "AARED"),
        help="Custom madd lengths (default: 4 4 4 4). Format: monfasel mottasel mottasel_waqf aared"
    )
    
    args = parser.parse_args()
    
    # Validate file output requirement
    if args.input_file and not args.output_file:
        parser.error("--output-file is required when using --input-file")
    
    # Create moshaf configuration
    madd_config = tuple(args.madd_config) if args.madd_config else None
    moshaf = create_moshaf(madd_config)
    
    # Process based on input mode
    if args.interactive:
        interactive_mode(moshaf, args.no_spaces)
    
    elif args.input_file:
        if not args.input_file.exists():
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        
        process_file(
            args.input_file,
            args.output_file,
            moshaf,
            args.no_spaces,
            not args.no_original
        )
    
    else:  # args.text
        result = process_text(
            args.text,
            moshaf,
            args.no_spaces,
            not args.no_original
        )
        print(result)
        try:
            phonetized_output = quran_phonetizer(args.text, moshaf, remove_spaces=args.no_spaces)
            print(f"Phonetized Output: {phonetized_output.phonemes}")
        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()

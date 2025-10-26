"""
ByT5 Phoneme-to-Uthmani Inference Module

This module provides a clean interface for converting phonemes to Uthmani script
using a trained ByT5 model. Can be used as a Python library or command-line tool.

Usage as Python Module:
    from infer_phoneme2uthmani_byt5 import PhonemeToUthmaniByT5
    
    # Initialize (automatically finds checkpoint)
    converter = PhonemeToUthmaniByT5()
    
    # Single conversion
    uthmani = converter("بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم")
    
    # Batch conversion
    uthmani_list = converter.batch_convert([phoneme1, phoneme2, ...])

Usage as Command Line:
    # Interactive mode
    python infer_phoneme2uthmani_byt5.py
    
    # Single conversion
    python infer_phoneme2uthmani_byt5.py --text "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم"
    
    # Batch file
    python infer_phoneme2uthmani_byt5.py --input phonemes.txt --output uthmani.txt
"""

import sys
import argparse
import logging
from pathlib import Path
import pathlib
from typing import Optional, List, Union

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PhonemeToUthmaniByT5:
    """
    ByT5-based phoneme-to-Uthmani text converter for inference.
    
    This class provides a simple interface for converting phoneme representations
    to Uthmani script using a trained ByT5 model.
    
    Attributes:
        device (str): Device to run inference on ('cuda', 'cpu', or 'mps')
        model: The loaded T5 model
        tokenizer: The tokenizer for the model
    
    Example:
        >>> converter = PhonemeToUthmaniByT5()
        >>> uthmani = converter("بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم")
        >>> print(uthmani)
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        base_model: str = "google/byt5-small",
    ):
        """
        Initialize the phoneme-to-Uthmani converter.
        
        Args:
            checkpoint_path: Path to fine-tuned checkpoint. If None, searches common locations.
            device: Device to use ('auto', 'cuda', 'mps', or 'cpu'). 'auto' selects best available.
            dtype: Data type for model (torch.float32, torch.bfloat16, etc.). None = auto-select.
            base_model: Base model name if checkpoint doesn't specify.
        """
        self.base_model = base_model
        self._setup_device(device, dtype)
        self._load_model(checkpoint_path)
        
    def _setup_device(self, device: str, dtype: Optional[torch.dtype]):
        """Determine and set up the compute device and dtype."""
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Determine dtype
        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif self.device == "cpu":
                self.dtype = torch.float32  # bfloat16 can work on CPU but float32 is safer
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype
        
        logger.info(f"Using device: {self.device}, dtype: {self.dtype}")
    
    def _find_checkpoint(self) -> Optional[Path]:
        """Search for checkpoint in common locations."""
        root = Path(__file__).parent
        
        # Search locations in priority order
        candidates = [
            root / "p2u_byt5_best.pt",
            root / "checkpoints" / "p2u_byt5_best.pt",
            root / "models" / "p2u_byt5_best.pt",
            root / "assets" / "p2u_byt5_best.pt",
            root / "p2u_byt5.pt",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found checkpoint: {candidate}")
                return candidate
        
        logger.warning("No checkpoint found in common locations. Using base model.")
        return None
    
    def _load_checkpoint_safe(self, checkpoint_path: Path) -> dict:
        """Load checkpoint with cross-platform path handling."""
        # Handle cross-platform path issues (Linux→Windows)
        if sys.platform == 'win32':
            # Create dummy pathlib._local module to handle corrupted checkpoints
            if 'pathlib._local' not in sys.modules:
                sys.modules['pathlib._local'] = type(sys)('pathlib._local')
                sys.modules['pathlib._local'].WindowsPath = pathlib.WindowsPath
            temp_posix = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        
        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            return ckpt
        finally:
            if sys.platform == 'win32':
                pathlib.PosixPath = temp_posix
    
    def _load_model(self, checkpoint_path: Optional[Union[str, Path]]):
        """Load model and tokenizer."""
        # Find checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()
        elif isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Load model
        logger.info(f"Loading base model: {self.base_model}")
        self.model = T5ForConditionalGeneration.from_pretrained(self.base_model)
        
        # Load fine-tuned weights if checkpoint exists
        checkpoint_loaded = False
        if checkpoint_path and checkpoint_path.exists():
            try:
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                ckpt = self._load_checkpoint_safe(checkpoint_path)
                
                # Extract model state
                if "model_state" in ckpt:
                    self.model.load_state_dict(ckpt["model_state"])
                    checkpoint_loaded = True
                    
                    # Log training info
                    if "epoch" in ckpt:
                        logger.info(f"  Epoch: {ckpt['epoch']}")
                    if "val_cer" in ckpt:
                        logger.info(f"  Validation CER: {ckpt['val_cer']:.4f}")
                else:
                    logger.warning("Checkpoint does not contain 'model_state'. Using base model.")
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Continuing with base model...")
        
        if not checkpoint_loaded:
            logger.warning("⚠️  Using base (untrained) model. Results may be poor.")
            logger.info("   Train a model first: python train_phoneme2uthmani_byt5.py")
        
        # Move to device and set dtype
        self.model = self.model.to(self.device)
        if self.dtype != torch.float32:
            try:
                self.model = self.model.to(dtype=self.dtype)
            except Exception as e:
                logger.warning(f"Could not convert to {self.dtype}: {e}")
        
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model ready. Parameters: {total_params:,}")
    
    @torch.inference_mode()
    def __call__(
        self,
        phonemes: str,
        max_length: int = 512,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Convert phonemes to Uthmani text.
        
        Args:
            phonemes: Input phoneme string
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search (1 = greedy)
            temperature: Sampling temperature (only if do_sample=True)
            do_sample: Use sampling instead of greedy/beam search
            
        Returns:
            Generated Uthmani text
        """
        if not phonemes or not phonemes.strip():
            return ""
        
        # Tokenize
        inputs = self.tokenizer(
            phonemes,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(self.device)
        
        # Generate
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
        
        try:
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return output.strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return ""
    
    def convert(self, phonemes: str, **kwargs) -> str:
        """Alias for __call__ with clearer name."""
        return self(phonemes, **kwargs)
    
    @torch.inference_mode()
    def batch_convert(
        self,
        phonemes_list: List[str],
        max_length: int = 512,
        num_beams: int = 1,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Convert multiple phoneme strings to Uthmani text.
        
        Args:
            phonemes_list: List of phoneme strings
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            batch_size: Number of samples to process at once
            show_progress: Show progress bar
            
        Returns:
            List of generated Uthmani texts
        """
        results = []
        
        # Try to use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(0, len(phonemes_list), batch_size),
                desc="Converting",
                disable=not show_progress
            )
        except ImportError:
            iterator = range(0, len(phonemes_list), batch_size)
        
        for i in iterator:
            batch = phonemes_list[i:i + batch_size]
            batch = [p if p else "" for p in batch]  # Handle None/empty
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            
            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False,
            )
            
            # Decode
            batch_results = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            results.extend([r.strip() for r in batch_results])
        
        return results


def interactive_mode(converter: PhonemeToUthmaniByT5):
    """Run the converter in interactive mode."""
    print("\n" + "=" * 80)
    print("Interactive ByT5 Phoneme-to-Uthmani Converter")
    print("=" * 80)
    print("Enter phoneme text to convert (or 'quit'/'exit' to stop)")
    print("Enter 'example' to see sample conversions")
    print("Enter 'beams N' to set beam search size (e.g., 'beams 5')")
    print("-" * 80)
    
    num_beams = 1
    
    # Example phonemes
    examples = [
        "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم",
        "ءَلحَمدُلِللَااهِرَببِلعَاالَمِۦۦۦۦن",
        "ءَررَحمَاانِررَحِۦۦۦۦم",
    ]
    
    while True:
        try:
            user_input = input("\n📝 Phonemes: ").strip()
            
            # Exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Change beam size
            if user_input.lower().startswith('beams '):
                try:
                    num_beams = int(user_input.split()[1])
                    print(f"✓ Beam size set to: {num_beams}")
                except (ValueError, IndexError):
                    print("❌ Invalid. Usage: beams 5")
                continue
            
            # Show examples
            if user_input.lower() == 'example':
                print("\n" + "=" * 80)
                print("Example Conversions:")
                print("=" * 80)
                for i, ex in enumerate(examples, 1):
                    uthmani = converter(ex, num_beams=num_beams)
                    print(f"\n{i}. Phonemes: {ex}")
                    print(f"   Uthmani:  {uthmani}")
                continue
            
            # Skip empty
            if not user_input:
                continue
            
            # Convert
            uthmani = converter(user_input, num_beams=num_beams)
            print(f"📖 Uthmani:  {uthmani}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def batch_file_mode(converter: PhonemeToUthmaniByT5, input_file: str, output_file: str, num_beams: int = 1):
    """Process phonemes from file and save results."""
    print(f"\n📂 Reading: {input_file}")
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        phonemes_list = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(phonemes_list)} phoneme strings")
    
    # Convert
    print(f"\n🔄 Converting (beams={num_beams})...")
    uthmani_list = converter.batch_convert(phonemes_list, num_beams=num_beams)
    
    # Save
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for phonemes, uthmani in zip(phonemes_list, uthmani_list):
            f.write(f"Phonemes: {phonemes}\n")
            f.write(f"Uthmani:  {uthmani}\n")
            f.write("-" * 80 + "\n")
    
    print(f"✓ Saved to: {output_file}")
    
    # Show samples
    print(f"\n📊 Sample results (first 3):")
    for i, (p, u) in enumerate(zip(phonemes_list[:3], uthmani_list[:3]), 1):
        print(f"\n{i}. {p[:60]}{'...' if len(p) > 60 else ''}")
        print(f"   → {u[:60]}{'...' if len(u) > 60 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description='ByT5 Phoneme-to-Uthmani Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python infer_phoneme2uthmani_byt5.py
  
  # Single conversion
  python infer_phoneme2uthmani_byt5.py --text "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم"
  
  # Batch file processing
  python infer_phoneme2uthmani_byt5.py --input phonemes.txt --output uthmani.txt
  
  # Use custom checkpoint
  python infer_phoneme2uthmani_byt5.py --checkpoint my_model.pt --text "..."
  
  # Use beam search for better quality
  python infer_phoneme2uthmani_byt5.py --text "..." --num-beams 5
        """
    )
    
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device: auto, cuda, cpu, mps (default: auto)')
    parser.add_argument('--text', type=str, help='Phoneme text to convert')
    parser.add_argument('--input', type=str, help='Input file (one phoneme per line)')
    parser.add_argument('--output', type=str, help='Output file for batch conversion')
    parser.add_argument('--num-beams', type=int, default=1,
                       help='Beam search size (1=greedy, 5=better quality, default: 1)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for file processing (default: 8)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("ByT5 Phoneme-to-Uthmani Converter")
    print("=" * 80)
    
    try:
        # Initialize converter
        converter = PhonemeToUthmaniByT5(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        
        # Determine mode
        if args.text:
            # Single conversion
            print(f"\n📝 Input Phonemes:")
            print(f"   {args.text}")
            
            uthmani = converter(args.text, num_beams=args.num_beams, max_length=args.max_length)
            
            print(f"\n📖 Output Uthmani:")
            print(f"   {uthmani}")
            
            if args.num_beams == 1:
                print(f"\n💡 Tip: Use --num-beams 5 for better quality")
            
        elif args.input:
            # Batch mode
            if not args.output:
                args.output = str(Path(args.input).with_suffix('')) + '_uthmani.txt'
            
            batch_file_mode(converter, args.input, args.output, args.num_beams)
            
        else:
            # Interactive mode
            interactive_mode(converter)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

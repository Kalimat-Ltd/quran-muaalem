# Quran Muaalem

<div align="center">
<strong>بعون الله وتوفيقه لا شريك له نقدم المعلم القرآني الذكي القادر على كشف أخطاء التلاوة والتجويد وصفات الحروف</strong>

[![PyPI][pypi-badge]][pypi-url]
[![Python Versions][python-badge]][python-url]
[![Hugging Face Model][hf-model-badge]][hf-model-url]
[![Hugging Face Dataset][hf-dataset-badge]][hf-dataset-url]
[![Google Colab][colab-badge]][colab-url]
[![arXiv][arxiv-badge]][arxiv-url]
[![MIT License][mit-badge]][mit-url]
[![Discord][discord-badge]][discord-url]

</div>

[pypi-badge]: https://img.shields.io/pypi/v/quran-muaalem.svg
[pypi-url]: https://pypi.org/project/quran-muaalem/
[mit-badge]: https://img.shields.io/github/license/obadx/quran-muaalem.svg
[mit-url]: https://github.com/obadx/quran-muaalem/blob/main/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/quran-muaalem.svg
[python-url]: https://pypi.org/project/quran-muaalem/
[colab-badge]: https://img.shields.io/badge/Google%20Colab-Open%20in%20Colab-F9AB00?logo=google-colab&logoColor=white
[colab-url]: https://colab.research.google.com/drive/1If0G9NtdXiSRu6PVGtIMvLwxizF2jspn?usp=sharing
[hf-model-badge]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue
[hf-model-url]: https://huggingface.co/obadx/muaalem-model-v3_0
[hf-dataset-badge]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange
[hf-dataset-url]: https://huggingface.co/datasets/obadx/muaalem-annotated-v3
[arxiv-badge]: https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg
[arxiv-url]: https://arxiv.org/abs/<PAPER_ID>
[discord-badge]: https://img.shields.io/badge/Discord-Join%20Community-7289da?logo=discord&logoColor=white
[discord-url]: https://discord.gg/hJWW6fCH

<div align="center" style="background-color: #f0f8ff; border-left: 5px solid #4CAF50; padding: 15px; margin: 20px 0; border-radius: 5px;">
  <h3 style="color: #2c3e50; margin-top: 0;">📖 رابط لتجربة المعلم القرآني</h3>
  <p style="margin: 10px 0;">يرجى الضغط على للتجربة:</p>
  <a href="https://662a040e1863a5445c.gradio.live" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 10px 0;">الرابط</a>
  <p style="background-color: #ffeb3b; padding: 8px; border-radius: 3px; display: inline-block; margin: 10px 0;">
    ⚠️ <strong>تنبيه:</strong> هذا الرابط سينتهي في <span style="color: #d32f2f; font-weight: bold;">27 أغسطس 2025</span>
  </p>
</div>

[![ALT_TEXT](https://img.youtube.com/vi/CsFoznO08-Q/0.jpg)](https://www.youtube.com/watch?v=CsFoznO08-Q)


## الممزيات

* مدرب على الرسم الصوتي للقرآن الكريم: [quran-transcript](https://github.com/obadx/quran-transcript) القادر على كشف أخطاء الحروف والتجويد وصفات الحروف
* نموذج معقول الحجم 660 MP 
* يحتاج فقط إله 1.5 GB من ذاكرة معالج الرسوميات
* معمارية مبتكرة: CTC متعدد المستويات

## المعمارية
معمارية مبتكرة: CTC متعدد المستويات. حيث كل مستوي يتدرب على وجه معين

![multi-lvel-ctc](./assets/figures/mutli-level-ctc.png)

## الخطوات المختصرة للتطوير

* تجميع التلاوت القرآنية من القراء المتقنين: [prepare-quran-dataset](https://github.com/obadx/prepare-quran-dataset)
* تقسيم التلاوت على حسب الوقف وليس الآية باستخدام [المقسم](https://github.com/obadx/recitations-segmenter)
* الحصو على النص القرآني من المقاطع الصوتية باسخدام [نموذج ترتيل](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
* تصحيح النصوص المستخرجة من ترتيل باستخدام  [خوارزمية التسميع](https://github.com/obadx/quran-transcript)
* تحويل الرسم الإملائي للرسم العثماني: [quran-transcript](https://github.com/obadx/quran-transcript)
* تحويل الرسم العثماني للرسم الصوتي للقرآني الكريم الذي يصف كل قواعد التجويد ما عدا الإشمام: [quran-transcript](https://github.com/obadx/quran-transcript)
* تدريب النموذج على معمارية [Wav2Vec2BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert)


## استخدام النوذج


### استخدام النموذج عن طريق واجهة gradio

قم بتزيل  [uv](https://docs.astral.sh/uv/) 

```bash
pip install uv
```
أو
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

بعد ذلك قم بتنزيل `ffmpeg`

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

أو من خلال `anaconda`
```bash
conda install ffmpeg
```

قم بتشغيل `gradio` ب command واحد فقط:
```bash
uvx --no-cache --from https://github.com/obadx/quran-muaalem.git[ui]  quran-muaalem-ui
```
او
```bash
uvx quran-muaalem[ui]  quran-muaalem-ui
```

### عن طريق python API


#### Installation

First, install the required dependencies:

```bash
# Install system dependencies
sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev

# Install Python packages
pip install quran-muaalem librosa "numba>=0.61.2"
```

## Basic Usage Example

```python
"""
Basic example of using the Quran Muaalem package for phonetic analysis of Quranic recitation.
"""

from dataclasses import asdict
import json
import logging

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
import torch
from librosa.core import load

# Import the main Muaalem class (adjust import based on your actual package structure)
from quran_muaalem import Muaalem

# Setup logging to see informative messages
logging.basicConfig(level=logging.INFO)

def analyze_recitation(audio_path):
    """
    Analyze a Quranic recitation audio file using the Muaalem model.
    
    Args:
        audio_path (str): Path to the audio file to analyze
    """
    # Configuration
    sampling_rate = 16000  # Must be 16000 Hz
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    
    # Step 1: Prepare the Quranic reference text
    # Get the Uthmani script for a specific verse (Aya 8, Surah 75 in this example)
    uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani
    
    # Step 2: Configure the recitation style (Moshaf attributes)
    moshaf = MoshafAttributes(
        rewaya="hafs",        # Recitation style (Hafs is most common)
        madd_monfasel_len=2,  # Length of separated elongation
        madd_mottasel_len=4,  # Length of connected elongation
        madd_mottasel_waqf=4, # Length of connected elongation when stopping
        madd_aared_len=2,     # Length of necessary elongation
    )
    # see: https://github.com/obadx/prepare-quran-dataset?tab=readme-ov-file#moshaf-attributes-docs
    
    # Step 3: Convert text to phonetic representation
    # see docs for phnetizer: https://github.com/obadx/quran-transcript
    phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)
    
    # Step 4: Initialize the Muaalem model
    muaalem = Muaalem(device=device)
    
    # Step 5: Load and prepare the audio
    wave, _ = load(audio_path, sr=sampling_rate, mono=True)
    
    # Step 6: Process the audio with the model
    # The model analyzes the phonetic properties of the recitation
    outs = muaalem(
        [wave],           # Audio data
        [phonetizer_out],          # Phonetic reference
        sampling_rate=sampling_rate
    )
    
    # Step 7: Display the results
    for out in outs:
        print("Predicted Phonemes:", out.phonemes.text)
        
        # Display detailed phonetic features for each phoneme
        for sifa in out.sifat:
            print(json.dumps(asdict(sifa), indent=2, ensure_ascii=False))
            print("*" * 30)
        print("-" * 40)

    # Explaining Results
    explain_for_terminal(
        outs[0].phonemes.text,
        phonetizer_out.phonemes,
        outs[0].sifat,
        phonetizer_out.sifat,
    )


if __name__ == "__main__":
    # Replace with the path to your audio file
    audio_path = "./assets/test.wav"
    
    try:
        analyze_recitation(audio_path)
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
```

Output:

```bash
ءِننننَللَااهَبِكُللِشَيءِنعَلِۦۦمُ۾۾۾بَرَااااءَتُممممِنَللَااهِوَرَسُۥۥلِه
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Phonemes ┃ Tafashie       ┃ Qalqla       ┃ Ghonna       ┃ Hams Or Jahr ┃ Safeer    ┃ Tikraar     ┃ Tafkheem Or Taqeeq ┃ Istitala      ┃ Shidda Or Rakhawa ┃ Itbaq    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ ءِ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ ننننَ     │ not_motafashie │ not_moqalqal │ maghnoon     │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ للَ       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ mofakham           │ not_mostateel │ between           │ monfateh │
│ اا       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ mofakham           │ not_mostateel │ rikhw             │ monfateh │
│ هَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ hams         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ بِ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ كُ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ hams         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ للِ       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ شَ        │ motafashie     │ not_moqalqal │ not_maghnoon │ hams         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ ي        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ ءِ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ ن        │ not_motafashie │ not_moqalqal │ maghnoon     │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ عَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ لِ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ ۦۦ       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ مُ        │ not_motafashie │ not_moqalqal │ maghnoon     │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ ۾۾۾      │ not_motafashie │ not_moqalqal │ maghnoon     │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ بَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ رَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ mokarar     │ mofakham           │ not_mostateel │ between           │ monfateh │
│ اااا     │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ mofakham           │ not_mostateel │ rikhw             │ monfateh │
│ ءَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ تُ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ hams         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ shadeed           │ monfateh │
│ ممممِ     │ not_motafashie │ not_moqalqal │ maghnoon     │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ نَ        │ not_motafashie │ not_moqalqal │ maghnoon     │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ للَ       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ mofakham           │ not_mostateel │ between           │ monfateh │
│ اا       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ mofakham           │ not_mostateel │ rikhw             │ monfateh │
│ هِ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ hams         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ وَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ رَ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ mokarar     │ mofakham           │ not_mostateel │ between           │ monfateh │
│ سُ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ hams         │ safeer    │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ ۥۥ       │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
│ لِ        │ not_motafashie │ not_moqalqal │ not_maghnoon │ jahr         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ between           │ monfateh │
│ ه        │ not_motafashie │ not_moqalqal │ not_maghnoon │ hams         │ no_safeer │ not_mokarar │ moraqaq            │ not_mostateel │ rikhw             │ monfateh │
└──────────┴────────────────┴──────────────┴──────────────┴──────────────┴───────────┴─────────────┴────────────────────┴───────────────┴───────────────────┴──────────┘
```

### API Docs

```python
class Muaalem:
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v3_2",
        device: str = "cpu",
        dtype=torch.bfloat16,
    ):
        """
        Initializing Muallem Model

        Args:
            model_name_or_path: the huggingface model name or path
            device: the device to run model on
            dtype: the torch dtype. Default is `torch.bfloat16` as the model was trained on
        """

    @torch.no_grad()
    def __call__(
        self,
        waves: list[list[float] | torch.FloatTensor | NDArray],
        ref_quran_phonetic_script_list: list[QuranPhoneticScriptOutput],
        sampling_rate: int,
    ) -> list[MuaalemOutput]:
        """Infrence Funcion for the Quran Muaalem Project

                waves: input waves  batch , seq_len with different formats described above
                ref_quran_phonetic_script_list (list[QuranPhoneticScriptOutput]): list of the
                    phonetized ouput of `quran_transcript.quran_phonetizer` with `remove_space=True`

                sampleing_rate (int): has to be 16000

        Returns:
            list[MuaalemOutput]:
                A list of output objects, each containing phoneme predictions and their
                phonetic features (sifat) for a processed input.

            Each MuaalemOutput contains:
                phonemes (Unit):
                    A dataclass representing the predicted phoneme sequence with:
                        text (str): Concatenated string of all phonemes.
                        probs (Union[torch.FloatTensor, list[float]]):
                            Confidence probabilities for each predicted phoneme.
                        ids (Union[torch.LongTensor, list[int]]):
                            Token IDs corresponding to each phoneme.

                sifat (list[Sifa]):
                    A list of phonetic feature dataclasses (one per phoneme) with the
                    following optional properties (each is a SingleUnit or None):
                        - phonemes_group (str): the phonemes associated with the `sifa`
                        - hams_or_jahr (SingleUnit): either `hams` or `jahr`
                        - shidda_or_rakhawa (SingleUnit): either `shadeed`, `between`, or `rikhw`
                        - tafkheem_or_taqeeq (SingleUnit): either `mofakham`, `moraqaq`, or `low_mofakham`
                        - itbaq (SingleUnit): either `monfateh`, or `motbaq`
                        - safeer (SingleUnit): either `safeer`, or `no_safeer`
                        - qalqla (SingleUnit): eithr `moqalqal`, or `not_moqalqal`
                        - tikraar (SingleUnit): either `mokarar` or `not_mokarar`
                        - tafashie (SingleUnit): either `motafashie`, or `not_motafashie`
                        - istitala (SingleUnit): either `mostateel`, or `not_mostateel`
                        - ghonna (SingleUnit): either `maghnoon`, or `not_maghnoon`

            Each SingleUnit in Sifa properties contains:
                text (str): The feature's categorical label (e.g., "hams", "shidda").
                prob (float): Confidence probability for this feature.
                idx (int): Identifier for the feature class.
        """
```

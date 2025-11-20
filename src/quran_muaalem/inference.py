import logging

from quran_transcript import chunck_phonemes, QuranPhoneticScriptOutput
from transformers import AutoFeatureExtractor
import torch
from numpy.typing import NDArray

from .modeling.multi_level_tokenizer import MultiLevelTokenizer
from .modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from .decode import (
    multilevel_greedy_decode,
    phonemes_level_greedy_decode,
)
from .muaalem_typing import Unit, SingleUnit, Sifa, MuaalemOutput

# Set up logging to inference.log
logging.basicConfig(
    filename='inference.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite the file each time
)


def format_sifat(
    level_to_units: dict[str, list[Unit]],
    chunked_phonemes_batch: list[list[str]],
    multi_level_tokenizer: MultiLevelTokenizer,
) -> list[list[Sifa]]:
    """Disabled - returns empty lists for sifat."""
    logging.debug("Skipping sifat formatting - returning empty lists")
    # Return empty list for each sequence in the batch
    return [[] for _ in chunked_phonemes_batch]


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
        logging.info(f"Initializing Muaalem model: {model_name_or_path} on device: {device} with dtype: {dtype}")
        self.device = device
        self.dtype = dtype

        self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
        self.multi_level_tokenizer = MultiLevelTokenizer(model_name_or_path)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        self.model.to(device, dtype=dtype)
        logging.info("Muaalem model initialized successfully")

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
        try:
            logging.info("Starting inference call")
            logging.info(f"Number of waves: {len(waves)}")
            logging.info(f"Sampling rate: {sampling_rate}")
            logging.info(f"Number of reference phonetic scripts: {len(ref_quran_phonetic_script_list)}")

            if sampling_rate != 16000:
                raise ValueError(f"`sampling_rate` has to be 16000 got: `{sampling_rate}`")

            # TODO: check input waves

            logging.info("Tokenizing reference phonetic scripts")
            # Tokanizing Ref
            level_to_ref_ids = self.multi_level_tokenizer.tokenize(
                [r.phonemes for r in ref_quran_phonetic_script_list],
                [r.sifat for r in ref_quran_phonetic_script_list],
                to_dict=True,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=1024,
            )["input_ids"]
            logging.info(f"Tokenized reference IDs shapes: { {k: v.shape for k, v in level_to_ref_ids.items()} }")

            logging.info("Processing audio features")
            features = self.processor(
                waves, sampling_rate=sampling_rate, return_tensors="pt"
            )
            logging.info(f"Audio features shapes: { {k: v.shape for k, v in features.items()} }")
            features = {k: v.to(self.device, dtype=self.dtype) for k, v in features.items()}
            logging.info("Moved features to device and dtype")

            logging.info("Running model inference")
            outs = self.model(**features, return_dict=False)[0]
            logging.info(f"Model outputs keys: {list(outs.keys())}")
            logging.info(f"Model outputs shapes: { {k: v.shape for k, v in outs.items()} }")

            logging.info("Computing softmax probabilities")
            probs = {}
            for level in outs:
                probs[level] = (
                    torch.nn.functional.softmax(outs[level], dim=-1).cpu().to(torch.float32)
                )
            logging.info(f"Probabilities shapes: { {k: v.shape for k, v in probs.items()} }")

            logging.info("Decoding phonemes level")
            # Decoding only Phonemes Level
            phonemes_units = phonemes_level_greedy_decode(
                probs["phonemes"], self.multi_level_tokenizer.id_to_vocab["phonemes"]
            )
            logging.info(f"Decoded {len(phonemes_units)} phoneme units")

            logging.info("Chunking phonemes")
            chunked_phonemes_batch: list[list[str]] = []
            for phonemes_unit in phonemes_units:
                chunked_phonemes_batch.append(chunck_phonemes(phonemes_unit.text))
            logging.info(f"Chunked phonemes batch lengths: {[len(chunks) for chunks in chunked_phonemes_batch]}")

            logging.info("Running multilevel greedy decode")
            level_to_units = multilevel_greedy_decode(
                level_to_probs=probs,
                level_to_id_to_vocab=self.multi_level_tokenizer.id_to_vocab,
                level_to_ref_ids=level_to_ref_ids,
                chunked_phonemes_batch=chunked_phonemes_batch,
                ref_chuncked_phonemes_batch=[
                    [s.phonemes for s in r.sifat] for r in ref_quran_phonetic_script_list
                ],
                phonemes_units=phonemes_units,
            )
            logging.info(f"Multilevel decode completed. Level to units keys: {list(level_to_units.keys())}")

            logging.info("Formatting sifat")
            sifat_batch: list[list[Sifa]] = format_sifat(
                level_to_units,
                chunked_phonemes_batch,
                self.multi_level_tokenizer,
            )
            logging.info(f"Formatted sifat batch lengths: {[len(sifat) for sifat in sifat_batch]}")

            logging.info("Creating output objects")
            outs = []
            # looping over the batch using phonemes_units directly
            for idx in range(len(phonemes_units)):
                outs.append(
                    MuaalemOutput(
                        phonemes=phonemes_units[idx],
                        sifat=sifat_batch[idx],
                    )
                )
            logging.info(f"Created {len(outs)} MuaalemOutput objects")
            logging.info("Inference call completed successfully")
            return outs
        except Exception as e:
            logging.error(f"Inference failed with error: {e}", exc_info=True)
            raise

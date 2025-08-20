from dataclasses import dataclass
from typing import Sequence


from quran_transcript import chunck_phonemes
from transformers import AutoFeatureExtractor
import torch
from numpy.typing import NDArray

from .modeling.multi_level_tokenizer import MultiLevelTokenizer
from .modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from .decode import multilevel_greedy_decode, align_sequence
from .muaalem_typing import Unit, SingleUnit, Sifa, MuaalemOutput, SingleUnit


def format_sifat(
    level_to_units: dict[str, list[Unit]],
    chunked_phonemes_batch: list[list[str]],
    multi_level_tokenizer: MultiLevelTokenizer,
    min_repeat=4,
) -> list[list[Sifa]]:
    sifat_batch = []
    for seq_idx in range(len(chunked_phonemes_batch)):
        sifat = []
        level_to_offset = {level: 0 for level in level_to_units}
        level_to_skip_ids = {
            level: align_sequence(
                level_to_units[level][seq_idx].ids,
                len(chunked_phonemes_batch[seq_idx]),
                min_repeat=min_repeat,
            )
            for level in level_to_units
        }
        for idx, ph_group in enumerate(chunked_phonemes_batch[seq_idx]):
            sifa_dict = {}
            for level in level_to_units:
                if level == "phonemes":
                    continue
                if idx in level_to_skip_ids[level]:
                    level_to_offset[level] = len(level_to_skip_ids[level])
                sifa_idx = idx + level_to_offset[level]
                if sifa_idx < len(level_to_units[level][seq_idx].ids):
                    label = int(level_to_units[level][seq_idx].ids[sifa_idx])
                    text = multi_level_tokenizer.sifat_to_en_vocab[level][label]
                    p = level_to_units[level][seq_idx].probs[sifa_idx]
                    sifa_dict[level] = SingleUnit(
                        text=text, prob=float(p), idx=int(label)
                    )
                else:
                    sifa_dict[level] = None
            sifat.append(
                Sifa(
                    phonemes_group=chunked_phonemes_batch[seq_idx][idx],
                    **sifa_dict,
                )
            )
        sifat_batch.append(sifat)
    return sifat_batch


class Muaalem:
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v2_1",
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
        self.device = device
        self.dtype = dtype

        self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
        self.multi_level_tokenizer = MultiLevelTokenizer(model_name_or_path)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        self.model.to(device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        waves: list[list[float] | torch.FloatTensor | NDArray],
        sampling_rate: int,
        min_repeat=4,
    ) -> list[MuaalemOutput]:
        if sampling_rate != 16000:
            raise ValueError(f"`sampling_rate` has to be 16000 got: `{sampling_rate}`")

        features = self.processor(
            waves, sampling_rate=sampling_rate, return_tensors="pt"
        )
        features = {k: v.to(self.device, dtype=self.dtype) for k, v in features.items()}
        outs = self.model(**features, return_dict=False)[0]

        probs = {}
        for level in outs:
            probs[level] = (
                torch.nn.functional.softmax(outs[level], dim=-1).cpu().to(torch.float32)
            )

        level_to_units = multilevel_greedy_decode(
            probs,
            self.multi_level_tokenizer.id_to_vocab,
        )

        chunked_phonemes_batch: list[list[str]] = []
        for phonemes_unit in level_to_units["phonemes"]:
            chunked_phonemes_batch.append(chunck_phonemes(phonemes_unit.text))

        sifat_batch: list[list[Sifa]] = format_sifat(
            level_to_units,
            chunked_phonemes_batch,
            self.multi_level_tokenizer,
            min_repeat=min_repeat,
        )

        outs = []
        # looping over the batch
        for idx in range(len(level_to_units["phonemes"])):
            outs.append(
                MuaalemOutput(
                    phonemes=level_to_units["phonemes"][idx],
                    sifat=sifat_batch[idx],
                )
            )
        return outs

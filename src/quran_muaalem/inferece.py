from dataclasses import dataclass
from typing import Sequence


from quran_transcript import chunck_phonemes
from transformers import AutoFeatureExtractor
import torch
from numpy.typing import NDArray

from .modeling.multi_level_tokenizer import MultiLevelTokenizer
from .modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from .decode import multilevel_greedy_decode
from .muaalem_typing import Unit, SingleUnit, Sifa, MuaalemOutput


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

        def __call__(
            self,
            waves: list[list[float] | torch.FloatTensor | NDArray],
            sampling_rate: int,
            top_n: int = 3,
        ) -> list[MuaalemOutput]:
            if sampling_rate != 16000:
                raise ValueError(
                    f"`sampling_rate` has to be 16000 got: `{sampling_rate}`"
                )

            features = self.processor(
                waves, sampling_rate=sampling_rate, return_tensors="pt"
            )
            features = {
                k: v.to(self.device, dtype=self.dtype) for k, v in features.items()
            }
            outs = self.model(**features, return_dict=False)[0]

            probs = {}
            for level in outs:
                probs[level] = (
                    torch.nn.functional.softmax(outs[level], dim=-1)
                    .cpu()
                    .to(torch.float32)
                )

            level_to_units: dict[str, Unit] = multilevel_greedy_decode(probs)

            chunked_phonemes_batch: list[list[str]] = []
            for phonemes_unit in level_to_units["phonemes"]:
                chunked_phonemes_batch.append(chunck_phonemes(phonemes_unit.text))

            sifat_batch: list[list[Sifa]] = self._fromat_sifat(
                level_to_units, chunked_phonemes_batch
            )

            outs = []
            for idx in range(len(level_to_units["phonemes"])):
                outs.append(
                    MuaalemOutput(
                        phonemes=level_to_units["phonemes"][idx],
                        sifat=sifat_batch[idx],
                    )
                )
            return outs

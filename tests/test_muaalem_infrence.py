from dataclasses import asdict
import json
from pathlib import Path

import torch

from quran_muaalem.inference import Muaalem
from quran_muaalem.muaalem_typing import MuaalemOutput
from torchcodec.decoders import AudioDecoder


def load_cache(cache_dir: str | Path, file_path: str | Path, reload=False):
    if reload:
        return
    file_path = Path(file_path)
    cache_path = Path(cache_dir) / f"{file_path.stem}.pt"
    if cache_path.is_file():
        print("Loading Cache")
        cache = torch.load(cache_path, weights_only=False)
        # outs = [MuaalemOutput(**item) for item in cache]
        # return outs
        return cache


def save_cache(cache_dir: str | Path, file_path: str | Path, outs: list[MuaalemOutput]):
    file_path = Path(file_path)
    cach_dir = Path(cache_dir)
    cach_dir.mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"{file_path.stem}.pt"
    # outs_dict = [asdict(o) for o in outs]
    torch.save(outs, cache_path)


if __name__ == "__main__":
    cache_dir = "./assets/test_cache"
    sampling_rate = 16000
    audio_path = "./assets/test.wav"

    cache = load_cache(cache_dir, audio_path, reload=False)
    if not cache:
        muaalem = Muaalem()
        decoder = AudioDecoder(audio_path, sample_rate=sampling_rate, num_channels=1)
        outs = muaalem(decoder.get_all_samples().data[0], sampling_rate)
        save_cache(cache_dir, audio_path, outs)
    else:
        outs = cache

    for out in outs:
        print(out.phonemes)
        for sifa in out.sifat:
            print(json.dumps(asdict(sifa), indent=2, ensure_ascii=False))
            print("*" * 30)
        print("-" * 40)

from dataclasses import asdict
import json

from quran_muaalem.inference import Muaalem
from torchcodec.decoders import AudioDecoder

if __name__ == "__main__":
    muaalem = Muaalem()
    sampling_rate = 16000

    decoder = AudioDecoder(
        "./assets/test.wav", sample_rate=sampling_rate, num_channels=1
    )
    outs = muaalem(decoder.get_all_samples().data[0], sampling_rate)

    for out in outs:
        print(out.phonemes)
        for sifa in out.sifat:
            print(json.dumps(asdict(sifa), indent=2, ensure_ascii=False))
            print("*" * 30)
        print("-" * 40)

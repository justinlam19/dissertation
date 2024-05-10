from copy import deepcopy
import gc

import numpy as np

from data.data import get_librispeech_data, random_choice
from extension.config.wav2vec2_config import wav2vec2_config
from extension.quantization import low_bit_benchmark

output_file_path = "output/extension_per_layer_quant.txt"

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)
np.seed(1337)
calibration_samples = random_choice(audios, 10)
n = 20
audio_subset = audios[:n]
ref_subset = references[:n]

asr_model, module_config = wav2vec2_config()

with open(output_file_path, "w+") as f:
    for module, submodules in module_config.items():
        for bits in range(8, 0, -1):
            m = deepcopy(asr_model)
            wer = low_bit_benchmark(
                model=m,
                modules=submodules,
                bits=bits,
                samples=audio_subset,
                references=ref_subset,
                calibration_samples=calibration_samples,
            )
            f.write(f"Module: {module}\nBits: {bits}\nWER:{wer}\n\n")
            del m
            gc.collect()

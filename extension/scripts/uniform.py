"""
A script for evaluating WER after uniform 4 bit quantization
"""

import sys

sys.path.append("/home/justinlam19/dissertation")


import gc
import itertools
from copy import deepcopy

import numpy as np

from data.data import get_librispeech_data, random_choice
from extension.config.wav2vec2_config import wav2vec2_config
from extension.quantization import low_bit_benchmark

output_file_path = "output/extension_uniform_4bit.txt"

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)
np.random.seed(1337)
calibration_samples = random_choice(audios, 10)
n = 20
audio_subset = audios[:n]
ref_subset = references[:n]

asr_model, module_config = wav2vec2_config()

with open(output_file_path, "w+") as f:
    m = deepcopy(asr_model)
    wer = low_bit_benchmark(
        model=m,
        modules=list(itertools.chain(module_config.values())),
        bits=4,
        samples=audio_subset,
        references=ref_subset,
        calibration_samples=calibration_samples,
    )
    f.write(f"Uniform 4 Bit Quantization\nWER(%):{wer}\n\n")
    del m
    gc.collect()

import sys

sys.path.append("/home/justinlam19/dissertation")

import gc
from copy import deepcopy

import numpy as np

from data.data import get_librispeech_data, random_choice
from extension.config.wav2vec2_config import wav2vec2_config
from extension.quantization import (
    calibrate,
    get_quant_modes,
    measure_wer,
    set_module_modes,
    wrap_modules,
)

output_file_path = "output/extension_overall.txt"

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)
np.seed(1337)
calibration_samples = random_choice(audios, 10)
n = 20
audio_subset = audios[:n]
ref_subset = references[:n]

model, module_config = wav2vec2_config()

m = deepcopy(model)
m.eval()
wer = measure_wer(
    model=m,
    samples=audio_subset,
    references=ref_subset,
)
with open(output_file_path, "a+") as f:
    f.write(f"Original model \nWER(%): {wer}\n\n")
del m
gc.collect()

m = deepcopy(model)
m.eval()
bits_config = {
    "encoder.enc": 5,
    "encoder.wav2vec2.model.encoder.layers": 6,
    "encoder.wav2vec2.model.feature_projection": 3,
    "encoder.wav2vec2.model.feature_extractor": 5,
}
quantize_weights = True
quantize_activations = False
quant_modes = get_quant_modes(quantize_weights, quantize_activations)
for module, bits in bits_config.items():
    wrap_modules(
        model=m,
        modules=[module],
        bits=bits,
        quantize_weights=quantize_weights,
        quantize_activations=quantize_activations,
    )

modules = []
for layers in module_config.values():
    modules += layers

set_module_modes(
    model=model,
    modules=modules,
    mode=quant_modes["calibration"],
)
calibrate(
    model=model,
    samples=calibration_samples,
)
set_module_modes(
    model=model,
    modules=modules,
    mode=quant_modes["evaluation"],
)
wer = measure_wer(
    model=model,
    samples=audio_subset,
    references=ref_subset,
)
with open(output_file_path, "a+") as f:
    f.write(
        f"Mixed resolution quantization (5 enc 3 proj 5 extractor 6 layers) \nWER(%): {wer}\n\n"
    )
del m
gc.collect()

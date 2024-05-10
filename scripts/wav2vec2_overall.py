import sys

sys.path.append("/home/justinlam19/dissertation")


import gc
from copy import deepcopy

import numpy as np

from benchmark.benchmark import benchmark
from config.config import ModelConfig, QuantMethod
from data.data import get_librispeech_data, random_choice
from quantization.quantization import custom_quantize

output_file = "output/wav2vec2_overall.txt"

model_config = ModelConfig.wav2vec2()
asr_model = model_config.type.from_hparams(
    source=model_config.src,
    savedir=model_config.savedir,
)

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)
np.seed(1337)
calibration_samples = random_choice(audios, 10)
n = 100
audio_subset = audios[:n]
ref_subset = references[:n]

original_model = deepcopy(asr_model)
original_model.eval()
wer, rtf = benchmark(original_model, audio_subset, ref_subset)
with open(output_file, "w+") as f:
    f.write(f"Original Model\nWER(%): {wer}\nRTF: {rtf}\n\n")
del original_model
gc.collect()

quantized_model = deepcopy(asr_model)
model_config.module_config["encoder.wav2vec2.model.feature_projection"] = [
    QuantMethod.STATIC
]
model_config.module_config["encoder.ctc_lin"] = []
dynamic_modules = [
    module
    for module in model_config.modules
    if QuantMethod.DYNAMIC in model_config.module_config[module]
]
static_modules = [
    module
    for module in model_config.modules
    if QuantMethod.STATIC in model_config.module_config[module]
]
custom_quantize(
    model=quantized_model,
    dynamic_modules=dynamic_modules,
    static_modules=static_modules,
    calibration_samples=calibration_samples,
)
quantized_model.eval()
wer, rtf = benchmark(quantized_model, audio_subset, ref_subset)
with open(output_file, "w+") as f:
    f.write(
        f"Quantized Model (dynamic enc, layers; static proj, extract)\nWER(%): {wer}\nRTF: {rtf}\n\n"
    )
del quantized_model
gc.collect()

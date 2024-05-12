"""
Script for experimenting with the effects of dynamic and static quantization.
Uses argparse and pickle to allow for other configurations to be passed in.
"""

import pickle
import sys

sys.path.append("/home/justinlam19/dissertation")

import argparse
import gc
from copy import deepcopy

import numpy as np

from benchmark.benchmark import benchmark
from config.config import ModelConfig, QuantMethod
from data.data import get_librispeech_data, random_choice
from quantization.quantization import custom_quantize

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="output file path")
parser.add_argument("-c", "--config", help="use preset config")
args = parser.parse_args()

output_file = args.output

if args.config == "wav2vec2":
    model_config = ModelConfig.wav2vec2()
elif args.config == "crdnn":
    model_config = ModelConfig.crdnn()
else:
    with open(args.config, "rb") as f:
        model_config = pickle.load(f)


asr_model = model_config.type.from_hparams(
    source=model_config.src,
    savedir=model_config.savedir,
)

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)
np.random.seed(1337)
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

for module in model_config.modules:
    if QuantMethod.DYNAMIC not in model_config.module_config[module]:
        continue
    quantized_model = deepcopy(asr_model)
    custom_quantize(
        model=quantized_model,
        dynamic_modules=[module],
        static_modules=None,
        calibration_samples=None,
    )
    quantized_model.eval()
    wer, rtf = benchmark(quantized_model, audio_subset, ref_subset)
    with open(output_file, "w+") as f:
        f.write(f"module (dynamic)\nWER(%): {wer}\nRTF: {rtf}\n\n")
    del quantized_model
    gc.collect()

for module in model_config.modules:
    if QuantMethod.STATIC not in model_config.module_config[module]:
        continue
    quantized_model = deepcopy(asr_model)
    custom_quantize(
        model=quantized_model,
        dynamic_modules=[module],
        static_modules=None,
        calibration_samples=None,
    )
    quantized_model.eval()
    wer, rtf = benchmark(quantized_model, audio_subset, ref_subset)
    with open(output_file, "w+") as f:
        f.write(f"module (static)\nWER(%): {wer}\nRTF: {rtf}\n\n")
    del quantized_model
    gc.collect()

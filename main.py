import gc
from copy import deepcopy

import numpy as np
from speechbrain.inference.ASR import EncoderASR

from benchmark.benchmark import benchmark
from quantization.quantization import custom_quantize
from utils.data import get_samples, random_choice

model_src = "speechbrain/asr-wav2vec2-commonvoice-14-en"
model_savedir = "pretrained_ASR/asr-wav2vec2-commonvoice-14-en"
output_file = "output/output.txt"

asr_model = EncoderASR.from_hparams(
    source=model_src,
    savedir=model_savedir,
)

audios, references = get_samples("/content/librispeech_dev_clean/LibriSpeech/dev-clean")
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

dynamic_modules = ["encoder.wav2vec2.model.encoder.layers", "encoder.enc"]
static_modules = [
    "encoder.wav2vec2.model.feature_projection",
    "encoder.wav2vec2.model.feature_extractor",
]
quantized_model = deepcopy(asr_model)
custom_quantize(
    model=quantized_model,
    dynamic_modules=dynamic_modules,
    static_modules=static_modules,
    calibration_samples=calibration_samples,
)
quantized_model.eval()
wer, rtf = benchmark(quantized_model, audio_subset, ref_subset)
with open(output_file, "w+") as f:
    f.write(f"Original Model\nWER(%): {wer}\nRTF: {rtf}\n\n")
del quantized_model
gc.collect()

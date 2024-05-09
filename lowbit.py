from copy import deepcopy
import gc
import numpy as np
from speechbrain.inference.ASR import EncoderASR

from extension.quantization import low_bit_benchmark
from quantization.utils import get_module, set_module
from data.data import get_librispeech_data, random_choice

model_src = "speechbrain/asr-wav2vec2-commonvoice-14-en"
model_savedir = "pretrained/asr-wav2vec2-commonvoice-14-en"
output_file = "output/extension.txt"

asr_model = EncoderASR.from_hparams(
    source=model_src,
    savedir=model_savedir,
)

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)

np.seed(1337)
calibration_samples = random_choice(audios, 10)

n = 20
audio_subset = audios[:n]
ref_subset = references[:n]

"""
m.eval()
w, rtf = benchmark(m, audios, references)
with open(output_file_path, "a+") as f:
    f.write(f"Original model \nwer: {w}\nrtf:{rtf}\n\n")
"""

for bits in range(8, 0, -1):
    m = deepcopy(asr_model)
    layers = []
    for _, layer in enumerate(get_module(m, "encoder.enc")):
        if "linear" in layer:
            layer_name = f"encoder.enc.{layer}.w"
            layers.append(layer_name)
    wer = low_bit_benchmark(
        model=m,
        modules=layers,
        bits=bits,
        samples=audio_subset,
        references=ref_subset,
        calibration_samples=calibration_samples,
    )
    del m
    gc.collect()

layer_name = "encoder.wav2vec2.model.feature_projection.projection"
layers.append(layer_name)
set_module(
    m,
    layer_name,
    QWrapper(
        get_module(m, layer_name),
        weight_quantizer=AffineQuantizer(3, BatchMinMax()),
        acts_quantizer=None,
    ),
)

for i in range(7):
    layer_name = f"encoder.wav2vec2.model.feature_extractor.conv_layers.{i}.conv"
    set_module(
        m,
        layer_name,
        QConv1dWrapper(
            get_module(m, layer_name),
            weight_quantizer=AffineQuantizer(5, BatchMinMax()),
            acts_quantizer=None,
        ),
    )

module = "encoder.wav2vec2.model.encoder.layers"
encoder_layers = []
for i in range(24):
    encoder_layers.append(f"{module}.{i}.attention.k_proj")
    encoder_layers.append(f"{module}.{i}.attention.v_proj")
    encoder_layers.append(f"{module}.{i}.attention.q_proj")
    encoder_layers.append(f"{module}.{i}.attention.out_proj")
    encoder_layers.append(f"{module}.{i}.feed_forward.intermediate_dense")
    encoder_layers.append(f"{module}.{i}.feed_forward.output_dense")
for layer_name in encoder_layers:
    set_module(
        m,
        layer_name,
        QWrapper(
            get_module(m, layer_name),
            weight_quantizer=AffineQuantizer(6, BatchMinMax()),
            acts_quantizer=None,
        ),
    )
layers += encoder_layers

m.eval()
for layer in layers:
    set_qmodule_state(get_module(m, layer), QModuleState.CALIBRATION_WEIGHT_ONLY)
_ = benchmark(m, audios[:20], references[:20])

for layer in layers:
    set_qmodule_state(get_module(m, layer), QModuleState.QUANT_EVAL_WEIGHT_ONLY)

w, rtf = benchmark(m, audios, references)
with open(output_file_path, "a+") as f:
    f.write(f"5 enc 3 proj 5 extractor 6 layers \nwer: {w}\nrtf:{rtf}\n\n")

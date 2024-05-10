from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR

from benchmark.flops import count_flops
from data.data import get_librispeech_data

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)

model_src = "speechbrain/asr-crdnn-commonvoice-14-en"
model_savedir = "pretrained/asr-crdnn-commonvoice-14-en"

asr_model = EncoderDecoderASR.from_hparams(
    source=model_src,
    savedir=model_savedir,
)

modules = [
    "encoder.model.RNN.rnn",
    "encoder.model.DNN",
    "decoder.dec",
    "decoder.fc.w",
    "encoder.model.CNN",
]


f = count_flops(asr_model, modules, audios[1])
print(f)

model_src = "speechbrain/asr-wav2vec2-commonvoice-14-en"
model_savedir = "pretrained/asr-wav2vec2-commonvoice-14-en"

asr_model = EncoderASR.from_hparams(
    source=model_src,
    savedir=model_savedir,
)

modules = [
    "encoder.wav2vec2.model.feature_projection",
    "encoder.wav2vec2.model.feature_extractor",
    "encoder.wav2vec2.model.encoder.layers",
    "encoder.enc",
    "encoder.ctc_lin",
]

f = count_flops(asr_model, modules, audios[1])
print(f)

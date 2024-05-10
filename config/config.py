from copy import deepcopy
from typing_extensions import Self

from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR


class ModelConfig:
    def __init__(self, src: str, savedir: str, modules:list[str], type: type):
        self.src = src
        self.savedir = savedir
        self.type = type
        self.modules = deepcopy(modules)

    @staticmethod
    def wav2vec2() -> Self:
        return ModelConfig(
            src="speechbrain/asr-wav2vec2-commonvoice-14-en",
            savedir="pretrained/asr-wav2vec2-commonvoice-14-en",
            modules=[
                "encoder.wav2vec2.model.feature_projection",
                "encoder.wav2vec2.model.feature_extractor",
                "encoder.wav2vec2.model.encoder.layers",
                "encoder.enc",
                "encoder.ctc_lin",
            ],
            type=EncoderASR,
        )
    
    @staticmethod
    def crdnn() -> Self:
        return ModelConfig(
            src="speechbrain/asr-crdnn-commonvoice-14-en",
            savedir="pretrained/asr-crdnn-commonvoice-14-en",
            modules=[
                "encoder.model.RNN.rnn",
                "encoder.model.DNN",
                "decoder.dec",
                "decoder.fc.w",
                "encoder.model.CNN",
            ],
            type=EncoderDecoderASR,
        )

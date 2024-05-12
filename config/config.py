from __future__ import annotations
from enum import Enum

from copy import deepcopy
from enum import Enum
from typing import Type

from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR, Pretrained


# Enum to indicate static/dynamic quantization for a given submodule
class QuantMethod(Enum):
    STATIC = 0
    DYNAMIC = 1


class ModelConfig:
    """
    Config of model, holds info about source path, saved directory, 
    valid quantization methods per module, and module type.

    wav2vec2 and crdnn (commonvoice-14-en) are pre-coded with the above information
    for ease of use.
    """
    def __init__(
        self,
        src: str,
        savedir: str,
        module_config: dict[str, list[QuantMethod]],
        type: Type[Pretrained],
    ):
        self.src = src
        self.savedir = savedir
        self.type = type
        self.module_config = deepcopy(module_config)
        self.modules = list(self.module_config.keys())

    @staticmethod
    def wav2vec2() -> ModelConfig:
        return ModelConfig(
            src="speechbrain/asr-wav2vec2-commonvoice-14-en",
            savedir="pretrained/asr-wav2vec2-commonvoice-14-en",
            module_config={
                "encoder.wav2vec2.model.feature_projection": [
                    QuantMethod.STATIC,
                    QuantMethod.DYNAMIC,
                ],
                "encoder.wav2vec2.model.feature_extractor": [QuantMethod.STATIC],
                "encoder.wav2vec2.model.encoder.layers": [QuantMethod.DYNAMIC],
                "encoder.enc": [QuantMethod.DYNAMIC],
                "encoder.ctc_lin": [QuantMethod.DYNAMIC],
            },
            type=EncoderASR,
        )

    @staticmethod
    def crdnn() -> ModelConfig:
        return ModelConfig(
            src="speechbrain/asr-crdnn-commonvoice-14-en",
            savedir="pretrained/asr-crdnn-commonvoice-14-en",
            module_config={
                "encoder.model.RNN.rnn": [QuantMethod.DYNAMIC],
                "encoder.model.DNN": [QuantMethod.DYNAMIC],
                "decoder.dec": [QuantMethod.DYNAMIC],
                "decoder.fc.w": [QuantMethod.STATIC, QuantMethod.DYNAMIC],
                "encoder.model.CNN": [QuantMethod.STATIC],
            },
            type=EncoderDecoderASR,
        )

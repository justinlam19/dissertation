"""
Functions for counting the number of FLOPs (per unit audio length),
so as to help with sensitivity analysis of quantization.
"""

import torch
from fvcore.nn import FlopCountAnalysis
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR


def count_flops(model, modules, sample):
    return _encoder_flop_analysis(model, modules, sample, FlopCountAnalysis)


# Pass dependency into function for easier testing
def _encoder_flop_analysis(model, modules, sample, flop_analyzer):
    if not isinstance(model, EncoderASR) and not isinstance(model, EncoderDecoderASR):
        raise NotImplementedError

    wavs = sample.unsqueeze(0).float()
    wav_lens = torch.tensor([1.0])

    # audio signal is 16kHz
    audio_length = sample.shape[0] / 16000

    flops = flop_analyzer(model.mods.encoder, (wavs, wav_lens))
    flops_by_module = flops.by_module()
    output = {}
    for module in modules:
        if module.startswith("encoder."):
            output[module] = (
                flops_by_module[module.removeprefix("encoder.")] / audio_length
            )

    return output

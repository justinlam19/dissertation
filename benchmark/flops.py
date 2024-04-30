import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from speechbrain.inference.ASR import EncoderASR

from benchmark.encoder_asr_wrapper import EncoderASRWrapper, generate, preprocess_input


def get_flops_per_unit_audio_length(model, modules, audio_sample):
    if isinstance(model, EncoderASR):
        wrapper = EncoderASRWrapper(model)
    else:
        raise NotImplementedError
    flops = FlopCountAnalysis(wrapper, audio_sample)
    flops_by_module = flops.by_module()
    output = {}
    for module in modules:
        output[module] = flops_by_module["mods." + module]
    return output

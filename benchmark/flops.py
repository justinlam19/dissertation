import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR

from benchmark.wrapper import EncoderASRWrapper, EncoderDecoderASRWrapper


"""
def count_flops(model, modules, audio_sample):
    if isinstance(model, EncoderASR):
        wrapper = EncoderASRWrapper(model)
    elif isinstance(model, EncoderDecoderASR):
        wrapper = EncoderDecoderASRWrapper(model)
    else:
        raise NotImplementedError
    flops = FlopCountAnalysis(wrapper, audio_sample)
    flops_by_module = flops.by_module()
    output = {}
    for module in modules:
        output[module] = flops_by_module["mods." + module]
    return output
"""

def count_flops(model, modules, sample):
    if not isinstance(model, EncoderASR) and not isinstance(model, EncoderDecoderASR):
        raise NotImplementedError
    
    wavs = sample.unsqueeze(0).float()
    wav_lens = torch.tensor([1.0])
    audio_length = sample.shape[0] / 16000

    flops = FlopCountAnalysis(model.mods.encoder, (wavs, wav_lens))
    flops_by_module = flops.by_module()
    output = {}
    for module in modules:
        if module.startswith("encoder."):
            output[module] = flops_by_module[module.removeprefix("encoder.")] / audio_length

    """    
    if not any(map(lambda x: x.startswith("decoder."), modules)):
        return output

    if isinstance(model, EncoderDecoderASR):
        encoder_out = model.mods.encoder(wavs, wav_lens)
        if model.transducer_beam_search:
            flops = FlopCountAnalysis(model.mods.decoder, encoder_out)
        else:
            flops = FlopCountAnalysis(model.mods.decoder, (encoder_out, wav_lens))      
        flops_by_module = flops.by_module()
        for module in modules:
            if module.startswith("decoder."):
                output[module] = flops_by_module[module.removeprefix("decoder.")] / audio_length
    """
            
    return output

import time

import torch
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR

from benchmark.encoder_asr import generate, preprocess_input


def timed_transcribe(model: EncoderASR, input):
    with torch.no_grad():
        wavs, wav_lens = preprocess_input(model, input)
        start = time.time()
        encoder_out = model.mods.encoder(wavs, wav_lens)
        end = time.time()
        duration = end - start
        predictions = model.decoding_function(encoder_out, wav_lens)
        predicted_words = generate(model, predictions)
    return predicted_words[0], duration

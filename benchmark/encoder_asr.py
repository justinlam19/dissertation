import functools
import speechbrain
import torch

from speechbrain.inference.ASR import EncoderASR


def preprocess_input(model: EncoderASR, input):
    with torch.no_grad():
        wavs = input.unsqueeze(0)
        wav_lens = torch.tensor([1.0])
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(model.device), wav_lens.to(model.device)
        return wavs, wav_lens


def generate(model: EncoderASR, predictions):
    is_ctc_text_encoder_tokenizer = isinstance(
        model.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder
    )
    if isinstance(model.hparams.decoding_function, functools.partial):
        if is_ctc_text_encoder_tokenizer:
            predicted_words = [
                "".join(model.tokenizer.decode_ndim(token_seq))
                for token_seq in predictions
            ]
        else:
            predicted_words = [
                model.tokenizer.decode_ids(token_seq) for token_seq in predictions
            ]
    else:
        predicted_words = [hyp[0].text for hyp in predictions]
    return predicted_words

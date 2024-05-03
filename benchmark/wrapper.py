import functools
import time

import speechbrain
import torch
import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]
        else:
            return getattr(self.__dict__["_modules"]["model"], name)  


class EncoderASRWrapper(Wrapper):
    def preprocess_input(self, input):
        with torch.no_grad():
            wavs = input.unsqueeze(0)
            wav_lens = torch.tensor([1.0])
            wavs = wavs.float()
            wavs, wav_lens = wavs.to(self.model.device), wav_lens.to(self.model.device)
        return wavs, wav_lens

    def generate(self, predictions):
        is_ctc_text_encoder_tokenizer = isinstance(
            self.model.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder
        )
        if isinstance(self.model.hparams.decoding_function, functools.partial):
            if is_ctc_text_encoder_tokenizer:
                predicted_words = [
                    "".join(self.model.tokenizer.decode_ndim(token_seq))
                    for token_seq in predictions
                ]
            else:
                predicted_words = [
                    self.model.tokenizer.decode_ids(token_seq)
                    for token_seq in predictions
                ]
        else:
            predicted_words = [hyp[0].text for hyp in predictions]
        return predicted_words

    def forward(self, input):
        with torch.no_grad():
            wavs, wav_lens = self.preprocess_input(input)
            encoder_out = self.model.mods.encoder(wavs, wav_lens)
            predictions = self.model.decoding_function(encoder_out, wav_lens)
            predicted_words = self.generate(predictions)
        return predicted_words[0]

    def timed_transcribe(self, input):
        with torch.no_grad():
            wavs, wav_lens = self.preprocess_input(input)
            start = time.time()
            encoder_out = self.model.mods.encoder(wavs, wav_lens)
            end = time.time()
            duration = end - start
            predictions = self.model.decoding_function(encoder_out, wav_lens)
            predicted_words = self.generate(predictions)
        return predicted_words[0], duration
    

class EncoderDecoderASRWrapper(Wrapper):
    def preprocess_input(self, input):
        with torch.no_grad():
            wavs = input.unsqueeze(0)
            wav_lens = torch.tensor([1.0])
            wavs = wavs.float()
            wavs, wav_lens = wavs.to(self.model.device), wav_lens.to(self.model.device)
        return wavs, wav_lens

    def generate(self, encoder_out, wav_lens):
        if self.model.transducer_beam_search:
            inputs = [encoder_out]
        else:
            inputs = [encoder_out, wav_lens]
        predicted_tokens, _, _, _ = self.model.mods.decoder(*inputs)
        predicted_words = [
            self.model.tokenizer.decode_ids(token_seq)
            for token_seq in predicted_tokens
        ]
        return predicted_words, predicted_tokens
    
    def forward(self, input):
        with torch.no_grad():
            wavs, wav_lens = self.preprocess_input(input)
            encoder_out = self.model.mods.encoder(wavs, wav_lens)
            predicted_words = self.generate(encoder_out, wav_lens)[0]
        return predicted_words[0]

    def timed_transcribe(self, input):
        with torch.no_grad():
            wavs, wav_lens = self.preprocess_input(input)
            start = time.time()
            encoder_out = self.model.mods.encoder(wavs, wav_lens)
            end = time.time()
            duration = end - start
            predicted_words = self.generate(encoder_out, wav_lens)[0]
        return predicted_words[0], duration

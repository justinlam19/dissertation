import tqdm
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR
from benchmark.wrapper import EncoderASRWrapper, EncoderDecoderASRWrapper
from benchmark.wer import compute_wer


def benchmark(model, samples, references):
    total_audio_length = sum([sample.shape[0] / 16000 for sample in samples])
    total_cpu_time = 0
    outputs = []

    if isinstance(model, EncoderASR):
        wrapper = EncoderASRWrapper(model)
    elif isinstance(model, EncoderDecoderASR):
        wrapper = EncoderDecoderASRWrapper(model)
    else:
        raise NotImplementedError

    for sample in tqdm.tqdm(samples[:10], desc="warming up"):
        wrapper.timed_transcribe(sample)

    for sample in tqdm.tqdm(samples, desc="evaluating"):
        output, duration = wrapper.timed_transcribe(sample)
        outputs.append(output)
        total_cpu_time += duration

    wer = compute_wer(references, outputs)
    rtf = total_cpu_time / total_audio_length
    return wer, rtf

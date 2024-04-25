import tqdm

from benchmark.rtf import timed_transcribe
from benchmark.wer import compute_wer


def benchmark(model, samples, references):
    total_audio_length = sum([sample.shape[0] / 16000 for sample in samples])
    total_cpu_time = 0
    outputs = []

    for sample in tqdm.tqdm(samples[:10], desc="warming up"):
        timed_transcribe(model, sample)

    for sample in tqdm.tqdm(samples, desc="evaluating"):
        output, duration = timed_transcribe(model, sample)
        outputs.append(output)
        total_cpu_time += duration

    wer = compute_wer(references, outputs)
    rtf = total_cpu_time / total_audio_length
    return wer, rtf

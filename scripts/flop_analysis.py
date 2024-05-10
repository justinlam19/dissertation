import sys
sys.path.append("/home/justinlam19/dissertation")

from benchmark.flops import count_flops
from config.config import ModelConfig
from data.data import get_librispeech_data

audios, references = get_librispeech_data("librispeech_dev_clean/LibriSpeech/dev-clean")
assert len(audios) == len(references)

def flop_analysis(model_config: ModelConfig, sample):
    asr_model = model_config.type.from_hparams(
        source=model_config.src,
        savedir=model_config.savedir,
    )
    return count_flops(asr_model, model_config.modules, sample)


print(flop_analysis(ModelConfig.wav2vec2(), audios[1]))
print(flop_analysis(ModelConfig.crdnn(), audios[1]))

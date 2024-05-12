"""
Config for extension, i.e. low bit quantization,
specifying the exact layers in the wav2vec2 model that can be quantized by TorchQuant
"""

from config.config import ModelConfig


def encoder_enc_config():
    module = "encoder.enc"
    layers = []
    for i in range(1, 4):
        layers.append(f"{module}.linear{i}.w")
    return layers


def feature_projection_config():
    module = "encoder.wav2vec2.model.feature_projection"
    return [module + ".projection"]


def feature_extractor_config():
    module = "encoder.wav2vec2.model.feature_extractor"
    layers = []
    for i in range(7):
        layers.append(f"{module}.conv_layers.{i}.conv")
    return layers


def encoder_layers_config():
    module = "encoder.wav2vec2.model.encoder.layers"
    layers = []
    for i in range(24):
        layers.append(f"{module}.{i}.attention.k_proj")
        layers.append(f"{module}.{i}.attention.v_proj")
        layers.append(f"{module}.{i}.attention.q_proj")
        layers.append(f"{module}.{i}.attention.out_proj")
        layers.append(f"{module}.{i}.feed_forward.intermediate_dense")
        layers.append(f"{module}.{i}.feed_forward.output_dense")
    return layers


def wav2vec2_config():
    model_config = ModelConfig.wav2vec2()
    model = model_config.type.from_hparams(
        source=model_config.src,
        savedir=model_config.savedir,
    )
    module_config = {
        "encoder.enc": encoder_enc_config(),
        "encoder.wav2vec2.model.encoder.layers": encoder_layers_config(),
        "encoder.wav2vec2.model.feature_projection": feature_projection_config(),
        "encoder.wav2vec2.model.feature_extractor": feature_extractor_config(),
    }
    return model, module_config

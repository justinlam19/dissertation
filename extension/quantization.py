import torch
from torchquant import set_qmodule_state, QModuleState, QWrapper
from torchquant.range_observers import ExpAvgMinMax
from torchquant.quantizers import AffineQuantizer

from benchmark.wer import compute_wer
from quantization.utils import get_module, set_module


def get_quant_modes(weight, act):
    modes = {
        "calibration": None,
        "evaluation": None,
    }
    if weight and act:
        modes["calibration"] = QModuleState.CALIBRATION
        modes["evaluation"] = QModuleState.QUANT_EVAL
    elif weight:
        modes["calibration"] = QModuleState.CALIBRATION_WEIGHT_ONLY
        modes["evaluation"] = QModuleState.QUANT_EVAL_WEIGHT_ONLY
    elif act:
        modes["calibration"] = QModuleState.CALIBRATION_ACT_ONLY
        modes["evaluation"] = QModuleState.QUANT_EVAL_ACT_ONLY
    else:
        raise Exception("At least one of weights or activations must be quantized")
    return modes


def wrap_modules(
    model,
    modules,
    bits,
    quantize_weights,
    quantize_activations,
):
    for module in modules:
        if quantize_weights:
            weight_quantizer = AffineQuantizer(bits, ExpAvgMinMax())
        else:
            weight_quantizer = None
        if quantize_activations:
            acts_quantizer = AffineQuantizer(bits, ExpAvgMinMax())
        else:
            acts_quantizer = None

        set_module(
            model,
            module,
            QWrapper(
                get_module(model, module),
                weight_quantizer=weight_quantizer,
                acts_quantizer=acts_quantizer,
            ),
        )


def set_module_modes(model, modules, mode):
    model.eval()
    for module in modules:
        set_qmodule_state(get_module(model, module), mode)


def calibrate(model, samples):
    for sample in samples:
        _ = model.transcribe_batch(sample.unsqueeze(0), torch.tensor([1.0]))


def measure_wer(model, samples, references):
    hypotheses = []
    for sample in samples:
        output = model.transcribe_batch(sample.unsqueeze(0), torch.tensor([1.0]))
        hypotheses.append(output)
    return compute_wer(references, hypotheses)


def low_bit_benchmark(
    model,
    modules,
    bits,
    samples,
    references,
    calibration_samples,
    quantize_weights=True,
    quantize_activations=False,
):
    quant_modes = get_quant_modes(quantize_weights, quantize_activations)
    wrap_modules(
        model=model,
        modules=modules,
        bits=bits,
        quantize_weights=quantize_weights,
        quantize_activations=quantize_activations,
    )
    set_module_modes(
        model=model,
        modules=modules,
        mode=quant_modes["calibration"],
    )
    calibrate(
        model=model,
        samples=calibration_samples,
    )
    set_module_modes(
        model=model,
        modules=modules,
        mode=quant_modes["evaluation"],
    )
    wer = measure_wer(
        model=model,
        samples=samples,
        references=references,
    )
    return wer

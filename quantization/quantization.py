"""
Custom function for quantizing a given model with a mix of quantization approaches.
"""

import torch
import torch.nn as nn

from quantization.static_quant import StaticQuant
from quantization.utils import get_module, set_module


def custom_quantize(
    model,
    dynamic_modules=None,
    static_modules=None,
    calibration_samples=None,
    dynamic_targets=None,
    dynamic_dtype=torch.qint8,
    static_qconfig=torch.ao.quantization.default_qconfig,
):
    """Performs in-place quantization of an ASR model

    The quantization is customizable. A combination of dynamic and static
    quantization can be performed on specific submodules that are passed into
    this function.

    Names of submodules passed into this class are implicitly assumed to be
    nested fields of ``model.mods``. For example, the ``model.mods.encoder.enc``
    submodule should be passed in as ``encoder.enc``.

    Reference https://pytorch.org/docs/stable/quantization.html for
    what torch modules can and cannot be dynamically/statically quantized.

    Arguments
    ---------
    model : torch.nn.Module
        Model to be quantized.
    dynamic_modules : list[str]
        Names of the submodules to be dynamically quantized. They should be
        formatted as stated above.
    static_modules : list[str]
        Names of the submodules to be statically quantized. They should be
        formatted as stated above.'
    calibration_samples : list[torch.Tensor]
        Sample inputs used for calibration during static quantization.
    dynamic_targets : set[torch.nn.Module]
        Torch modules to be quantized during dynamic quantization.
    dynamic_dtype : torch.dtype
        The torch datatype that values will be converted to during dynamic
        quantization. This should be a quantized datatype, such as
        ``torch.quint8``, ``torch.qint8``, ``torch.qint32``
    static_qconfig : torch.ao.quantization.qconfig.QConfig
        The quantization config for static quantization, which, among other
        things, specifies the observer modules that will be inserted
        and the resolution of quantization.

    Returns
    -------
    None
    """

    ##################################################
    # Dynamic Quantization                           #
    ##################################################
    dynamic_quantize(
        model=model,
        modules=dynamic_modules,
        targets=dynamic_targets,
        dtype=dynamic_dtype,
        quantize_fn=torch.quantization.quantize_dynamic,
    )

    ##################################################
    # Static Quantization                            #
    ##################################################
    static_quantize(
        model=model,
        modules=static_modules,
        calibration_samples=calibration_samples,
        qconfig=static_qconfig,
        prepare_fn=torch.ao.quantization.prepare,
        convert_fn=torch.ao.quantization.convert,
    )


# Get quantize_fn as parameter so the dependency can be mocked
def dynamic_quantize(model, modules, targets, dtype, quantize_fn):
    if modules is not None and len(modules) > 0:
        if targets is None:
            targets = {
                nn.LSTM,
                nn.GRU,
                nn.RNNCell,
                nn.GRUCell,
                nn.LSTMCell,
                nn.Linear,
            }

        for module in modules:
            quantize_fn(
                model=get_module(model, module),
                qconfig_spec=targets,
                dtype=dtype,
                inplace=True,
            )


# Get prepare_fn, convert_fn as parameters so the dependencies can be mocked
def static_quantize(
    model, modules, calibration_samples, qconfig, prepare_fn, convert_fn
):
    if modules is not None and len(modules) > 0:
        if calibration_samples is None or len(calibration_samples) == 0:
            raise Exception("No calibration samples provided for static quantization.")

        for module in modules:
            set_module(
                model,
                module,
                StaticQuant(get_module(model, module)),
            )
            get_module(model, module).qconfig = qconfig

        prepare_fn(model=model, inplace=True)

        for sample in calibration_samples:
            model.transcribe_batch(sample.unsqueeze(0), torch.tensor([1.0]))

        convert_fn(module=model, inplace=True)

import torch

from quantization.static_quant import StaticQuant
from quantization.utils import get_module, set_module


def custom_quantize(
    model,
    dynamic_modules,
    static_modules,
    calibration_samples,
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
    if dynamic_targets is None:
        dynamic_targets = {
            torch.nn.LSTM,
            torch.nn.GRU,
            torch.nn.RNNCell,
            torch.nn.GRUCell,
            torch.nn.LSTMCell,
            torch.nn.Linear,
        }

    for module in dynamic_modules:
        set_module(
            model,
            module,
            torch.quantization.quantize_dynamic(
                get_module(model, module),
                dynamic_targets,
                dtype=dynamic_dtype,
            ),
        )

    ##################################################
    # Static Quantization                            #
    ##################################################
    for module in static_modules:
        set_module(
            model,
            module,
            StaticQuant(get_module(model, module)),
        )
        get_module(model, module).qconfig = static_qconfig

    torch.ao.quantization.prepare(model, inplace=True)

    for sample in calibration_samples:
        model.transcribe_batch(sample.unsqueeze(0), torch.tensor([1.0]))

    torch.ao.quantization.convert(model, inplace=True)

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
from torchquant import QModuleState, QWrapper
from torchquant.quantizers import AffineQuantizer
from torchquant.range_observers import ExpAvgMinMax
import torch.nn as nn

from extension.quantization import get_quant_modes, wrap_modules


class TestGetQuantModes:
    def test_invalid_input(self):
        # GIVEN
        #      neither weights nor activations are marked for quantization
        weight = False
        act = False
        error = "At least one of weights or activations must be quantized"

        # WHEN
        #      the quantization modes are determined
        with pytest.raises(Exception) as exc_info:
            _ = get_quant_modes(weight, act)

        # THEN
        #      an error about needing weights and/or acts to be quantized is raised
        assert str(exc_info.value) == error

    @pytest.mark.parametrize(
        "weight,act,expected_calibration,expected_evaluation",
        [
            (True, True, QModuleState.CALIBRATION, QModuleState.QUANT_EVAL),
            (
                True,
                False,
                QModuleState.CALIBRATION_WEIGHT_ONLY,
                QModuleState.QUANT_EVAL_WEIGHT_ONLY,
            ),
            (
                False,
                True,
                QModuleState.CALIBRATION_ACT_ONLY,
                QModuleState.QUANT_EVAL_ACT_ONLY,
            ),
        ],
    )
    def test_all_valid_combinations(
        self, weight, act, expected_calibration, expected_evaluation
    ):
        # GIVEN
        #      different valid combinations of whether weights and activations are quantized
        # WHEN
        #      the quantization modes are determined
        modes = get_quant_modes(weight, act)

        # THEN
        #      the returned quantization modes are correct
        assert modes["calibration"] == expected_calibration
        assert modes["evaluation"] == expected_evaluation


class TestWrapModules:
    @pytest.mark.parametrize(
        "quantize_weights,quantize_activations",
        [(True, True), (True, False), (False, True)],
    )
    def test_all_combinations(self, quantize_weights, quantize_activations):
        # GIVEN
        #      model, modules, and number of bits are set
        #      various combinations of weight and activation quantization
        model = MagicMock()
        modules = ["module1", "module2"]
        model.mods = nn.Sequential(OrderedDict([
            (modules[0], nn.Linear(1, 3)),
            (modules[1], nn.Linear(3, 1)),
        ]))
        bits = 4

        # WHEN
        #      the modules are wrapped
        wrap_modules(
            model=model,
            modules=modules,
            bits=bits,
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
        )

        # THEN
        #      the specified modules are wrapped with QWrapper
        #      the wrappers have the expected attributes
        for module in (model.mods.module1, model.mods.module2):
            assert isinstance(module, QWrapper)
            if quantize_weights:
                assert isinstance(module.weight_quantizer, AffineQuantizer)
                assert module.weight_quantizer.n_bits == bits
                assert isinstance(module.weight_quantizer.observer, ExpAvgMinMax)
            else:
                assert module.weight_quantizer is None
            if quantize_activations:
                assert isinstance(module.acts_quantizer, AffineQuantizer)
                assert module.acts_quantizer.n_bits == bits
                assert isinstance(module.acts_quantizer.observer, ExpAvgMinMax)
            else:
                assert module.acts_quantizer is None




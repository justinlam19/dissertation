from collections import OrderedDict
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from torchquant import QModuleState, QWrapper
from torchquant.quantizers import AffineQuantizer
from torchquant.range_observers import ExpAvgMinMax

from extension.quantization import (calibrate, get_quant_modes,
                                    low_bit_benchmark, measure_wer,
                                    wrap_modules)


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
        model.mods = nn.Sequential(
            OrderedDict(
                [
                    (modules[0], nn.Linear(1, 3)),
                    (modules[1], nn.Linear(3, 1)),
                ]
            )
        )
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


class TestCalibrate:
    def test_calibrate(self):
        # GIVEN
        #      model, modules, and mode are specified
        #      modules are QWrapped
        model = MagicMock()
        model.transcribe_batch = MagicMock()
        modules = ["module1", "module2"]
        model.mods = nn.Sequential(
            OrderedDict(
                [
                    (modules[0], nn.Linear(1, 3)),
                    (modules[1], nn.Linear(3, 1)),
                ]
            )
        )
        wrap_modules(
            model=model,
            modules=modules,
            bits=4,
            quantize_weights=True,
            quantize_activations=False,
        )
        mode = QModuleState.CALIBRATION

        mock_sample1_unsqueeze = 42
        mock_sample2_unsqueeze = 1337
        sample1 = MagicMock()
        sample1.unsqueeze.return_value = mock_sample1_unsqueeze
        sample2 = MagicMock()
        sample2.unsqueeze.return_value = mock_sample2_unsqueeze
        samples = [sample1, sample2]
        expected_transcribe_calls = [
            call(mock_sample1_unsqueeze, torch.tensor([1.0])),
            call(mock_sample2_unsqueeze, torch.tensor([1.0])),
        ]

        # WHEN
        #      model is calibrated
        calibrate(model=model, modules=modules, mode=mode, samples=samples)

        # THEN
        #      modules are correctly set to the provided mode
        #      transcribe_batch is given correct inputs
        assert model.mods.module1.mode == mode
        assert model.mods.module2.mode == mode
        model.transcribe_batch.assert_has_calls(expected_transcribe_calls)


class TestMeasureWER:
    def test_measure_wer(self):
        # GIVEN
        #      model, modules, and mode are specified
        #      modules are QWrapped
        model = MagicMock()
        modules = ["module1", "module2"]
        model.mods = nn.Sequential(
            OrderedDict(
                [
                    (modules[0], nn.Linear(1, 3)),
                    (modules[1], nn.Linear(3, 1)),
                ]
            )
        )
        wrap_modules(
            model=model,
            modules=modules,
            bits=4,
            quantize_weights=True,
            quantize_activations=False,
        )
        mode = QModuleState.QUANT_EVAL
        samples = [MagicMock(), MagicMock()]
        references = ["reference one", "reference two"]
        model.transcribe_batch = MagicMock()
        model.transcribe_batch.return_value = "reference"
        expected_wer = 50.0

        # WHEN
        #      model wer is measured
        wer = measure_wer(
            model=model,
            modules=modules,
            mode=mode,
            samples=samples,
            references=references,
        )

        # THEN
        #      modules are correctly set to the provided mode
        #      correct WER is computed
        assert model.mods.module1.mode == mode
        assert model.mods.module2.mode == mode
        assert wer == pytest.approx(expected_wer)


class TestLowBitBenchmark:
    def test_low_bit_benchmark(self):
        # GIVEN
        #      all inputs are specified correctly
        model = MagicMock()
        modules = ["module1", "module2"]
        model.mods = nn.Sequential(
            OrderedDict(
                [
                    (modules[0], nn.Linear(1, 3)),
                    (modules[1], nn.Linear(3, 1)),
                ]
            )
        )
        bits = 4
        calibration_samples = [MagicMock()]
        samples = [MagicMock(), MagicMock()]
        references = ["reference one", "reference two"]
        model.transcribe_batch = MagicMock()
        model.transcribe_batch.return_value = "reference"
        expected_wer = 50.0

        # WHEN
        #      the performance of a low bit quantized model is benchmarked
        wer = low_bit_benchmark(
            model=model,
            modules=modules,
            bits=bits,
            samples=samples,
            references=references,
            calibration_samples=calibration_samples,
            quantize_weights=True,
            quantize_activations=False,
        )

        # THEN
        #      no errors
        #      the correct WER is computed
        assert wer == pytest.approx(expected_wer)

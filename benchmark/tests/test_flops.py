from unittest.mock import MagicMock

import pytest
import torch
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR

from benchmark.flops import _encoder_flop_analysis, count_flops


class TestCountFlops:
    def test_not_implemented(self):
        # GIVEN
        #      function not implemented for model type
        model = MagicMock()
        modules = ["encoder.module1", "encoder.module2"]
        sample = torch.Tensor([0.0])

        # WHEN
        #      flop analysis is carried out
        # THEN
        #      NotImplementedError is raised
        with pytest.raises(NotImplementedError):
            count_flops(model, modules, sample)


class TestEncoderFlopAnalysis:
    def test_encoder_flop_analysis(self):
        # GIVEN
        #      model is valid
        #      sample and modules are specified
        #      flopcount analysis object is provided
        model = MagicMock(spec=EncoderASR)
        modules = ["encoder.module1", "encoder.module2"]
        sample = torch.full((1600, 5), 1.0)
        model.mods = MagicMock()
        model.mods.encoder = MagicMock()

        flop_values = {
            "module1": 100,
            "module2": 200,
        }
        flop_analyzer = MagicMock()
        flop_object = MagicMock()
        flop_object.by_module = MagicMock()
        flop_object.by_module.return_value = flop_values
        flop_analyzer.return_value = flop_object

        audio_length = sample.shape[0] / 16000
        expected_output = {
            "encoder.module1": flop_values["module1"] / audio_length,
            "encoder.module2": flop_values["module2"] / audio_length,
        }

        # WHEN
        #      encoder flop analysis is carried out
        output = _encoder_flop_analysis(model, modules, sample, flop_analyzer)

        # THEN
        #      no error
        #      correct methods are called once each
        #      flop per unit audio length is calculated as expected
        flop_analyzer.assert_called_once()
        flop_object.by_module.assert_called_once()
        assert output == expected_output

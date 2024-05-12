from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn

from quantization.quantization import dynamic_quantize, static_quantize
from quantization.static_quant import StaticQuant
from quantization.utils import get_module


class TestDynamicQuantize:
    def test_dynamic_quantize_with_targets(self):
        # GIVEN
        #      the target nn.Modules types are specified
        #      all other inputs are also specified
        quantize_fn = MagicMock()
        model = MagicMock()
        targets = {"target1", "target2"}
        dtype = int
        modules = ["module1", "module2"]
        expected_calls = [
            call(
                model=get_module(model, module),
                qconfig_spec=targets,
                dtype=dtype,
                inplace=True,
            )
            for module in modules
        ]

        # WHEN
        #      dynamic quantization is applied
        dynamic_quantize(
            model=model,
            modules=modules,
            targets=targets,
            dtype=dtype,
            quantize_fn=quantize_fn,
        )

        # THEN
        #      the inner quantization function is called with the provided inputs
        quantize_fn.assert_has_calls(expected_calls)

    def test_dynamic_quantize_without_targets(self):
        # GIVEN
        #      the target nn.Modules types are not specified
        #      all other inputs are specified
        quantize_fn = MagicMock()
        model = MagicMock()
        dtype = int
        modules = ["module1", "module2"]
        targets = None
        expected_targets = {
            nn.LSTM,
            nn.GRU,
            nn.RNNCell,
            nn.GRUCell,
            nn.LSTMCell,
            nn.Linear,
        }
        expected_calls = [
            call(
                model=get_module(model, module),
                qconfig_spec=expected_targets,
                dtype=dtype,
                inplace=True,
            )
            for module in modules
        ]

        # WHEN
        #      dynamic quantization is applied
        dynamic_quantize(
            model=model,
            modules=modules,
            targets=targets,
            dtype=dtype,
            quantize_fn=quantize_fn,
        )

        # THEN
        #      the inner quantization function is called with the provided inputs
        #      the inner quantization function is called with the default targets
        quantize_fn.assert_has_calls(expected_calls)

    def test_dynamic_quantize_with_empty_module_list(self):
        # GIVEN
        #      the modules list is empty
        #      all other inputs are specified
        quantize_fn = MagicMock()
        model = MagicMock()
        targets = {"target1", "target2"}
        dtype = int
        modules = []

        # WHEN
        #      dynamic quantization is applied
        dynamic_quantize(
            model=model,
            modules=modules,
            targets=targets,
            dtype=dtype,
            quantize_fn=quantize_fn,
        )

        # THEN
        #      the inner quantization function is not called
        quantize_fn.assert_not_called()

    def test_dynamic_quantize_without_modules(self):
        # GIVEN
        #      the modules are None
        #      all other inputs are specified
        quantize_fn = MagicMock()
        model = MagicMock()
        targets = {"target1", "target2"}
        dtype = int
        modules = None

        # WHEN
        #      dynamic quantization is applied
        dynamic_quantize(
            model=model,
            modules=modules,
            targets=targets,
            dtype=dtype,
            quantize_fn=quantize_fn,
        )

        # THEN
        #      the inner quantization function is not called
        quantize_fn.assert_not_called()


class TestStaticQuantize:
    def test_calibration_samples_is_none(self):
        # GIVEN
        #      calibration samples is None
        #      there are modules to statically quantize
        #      all other inputs are specified
        prepare_fn = MagicMock()
        convert_fn = MagicMock()
        model = MagicMock()
        modules = ["module1", "module2"]
        calibration_samples = None
        qconfig = "qconfig"
        expected_error = "No calibration samples provided for static quantization."

        # WHEN
        #      static quantization is applied
        with pytest.raises(Exception) as exc_info:
            static_quantize(
                model=model,
                modules=modules,
                calibration_samples=calibration_samples,
                qconfig=qconfig,
                prepare_fn=prepare_fn,
                convert_fn=convert_fn,
            )

        # THEN
        #      An exception is thrown for no calibration samples
        assert str(exc_info.value) == expected_error

    def test_calibration_samples_is_empty(self):
        # GIVEN
        #      the calibration samples is empty
        #      there are modules to statically quantize
        #      all other inputs are specified
        prepare_fn = MagicMock()
        convert_fn = MagicMock()
        model = MagicMock()
        modules = ["module1", "module2"]
        calibration_samples = []
        qconfig = "qconfig"
        expected_error = "No calibration samples provided for static quantization."

        # WHEN
        #      static quantization is applied
        with pytest.raises(Exception) as exc_info:
            static_quantize(
                model=model,
                modules=modules,
                calibration_samples=calibration_samples,
                qconfig=qconfig,
                prepare_fn=prepare_fn,
                convert_fn=convert_fn,
            )

        # THEN
        #      An exception is thrown for no calibration samples
        assert str(exc_info.value) == expected_error

    def test_modules_is_none(self):
        # GIVEN
        #      the modules list for static quantization is None
        #      all other inputs are specified
        prepare_fn = MagicMock()
        convert_fn = MagicMock()
        model = MagicMock()
        modules = None
        calibration_samples = [MagicMock(), MagicMock()]
        qconfig = "qconfig"

        # WHEN
        #      static quantization is applied
        static_quantize(
            model=model,
            modules=modules,
            calibration_samples=calibration_samples,
            qconfig=qconfig,
            prepare_fn=prepare_fn,
            convert_fn=convert_fn,
        )

        # THEN
        #      The prepare and convert functions are not called
        #      The model is not calibrated
        prepare_fn.assert_not_called()
        convert_fn.assert_not_called()
        model.transcribe_batch.assert_not_called()

    def test_modules_is_empty(self):
        # GIVEN
        #      the modules list for static quantization is empty
        #      all other inputs are specified
        prepare_fn = MagicMock()
        convert_fn = MagicMock()
        model = MagicMock()
        modules = []
        calibration_samples = [MagicMock(), MagicMock()]
        qconfig = "qconfig"

        # WHEN
        #      static quantization is applied
        static_quantize(
            model=model,
            modules=modules,
            calibration_samples=calibration_samples,
            qconfig=qconfig,
            prepare_fn=prepare_fn,
            convert_fn=convert_fn,
        )

        # THEN
        #      The prepare and convert functions are not called
        #      The model is not calibrated
        prepare_fn.assert_not_called()
        convert_fn.assert_not_called()
        model.transcribe_batch.assert_not_called()

    def test_static_quantization(self):
        # GIVEN
        #      the modules list for static quantization is empty
        #      all other inputs are specified
        prepare_fn = MagicMock()
        convert_fn = MagicMock()
        model = MagicMock()
        modules = ["module1", "module2"]

        mock_sample1_unsqueeze = 42
        mock_sample2_unsqueeze = 1337
        sample1 = MagicMock()
        sample1.unsqueeze.return_value = mock_sample1_unsqueeze
        sample2 = MagicMock()
        sample2.unsqueeze.return_value = mock_sample2_unsqueeze
        calibration_samples = [sample1, sample2]
        qconfig = "qconfig"
        expected_prepare_calls = [call(model=model, inplace=True)]
        expected_convert_calls = [call(module=model, inplace=True)]
        expeceted_transcribe_calls = [
            call(mock_sample1_unsqueeze, torch.tensor([1.0])),
            call(mock_sample2_unsqueeze, torch.tensor([1.0])),
        ]

        # WHEN
        #      static quantization is applied
        static_quantize(
            model=model,
            modules=modules,
            calibration_samples=calibration_samples,
            qconfig=qconfig,
            prepare_fn=prepare_fn,
            convert_fn=convert_fn,
        )

        # THEN
        #      The specified modules are wrapped with StaticQuant
        #      The specified modules' qconfig is set
        #      The prepare and convert functions are called with correct inputs
        #      The model is calibrated with the correct inputs
        assert isinstance(model.mods.module1, StaticQuant)
        assert isinstance(model.mods.module2, StaticQuant)
        assert model.mods.module1.qconfig == qconfig
        assert model.mods.module2.qconfig == qconfig
        prepare_fn.assert_has_calls(expected_prepare_calls)
        convert_fn.assert_has_calls(expected_convert_calls)
        model.transcribe_batch.assert_has_calls(expeceted_transcribe_calls)

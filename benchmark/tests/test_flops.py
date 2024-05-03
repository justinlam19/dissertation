from unittest.mock import MagicMock

import pytest
import torch

from benchmark.flops import count_flops


class TestFLOPs:
    def test_not_EncoderASR(self):
        # GIVEN
        #      wrapper is not implemented for model type
        model = MagicMock()
        modules = ["module1", "module2"]
        sample = torch.Tensor([0.0])

        # WHEN
        #      flop analysis is carried out
        # THEN
        #      NotImplementedError is raised
        with pytest.raises(NotImplementedError):
            count_flops(model, modules, sample)

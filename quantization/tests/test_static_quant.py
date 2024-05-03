from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub

from quantization.static_quant import StaticQuant


class TestStaticQuantInit:
    def test_init(self):
        # GIVEN
        #      model is provided
        model = MagicMock(spec=nn.Module)

        # WHEN
        #      StaticQuant object is initialized with model
        wrapper = StaticQuant(model)

        # THEN
        #      StaticQuant object is initialized with correct fields
        assert hasattr(wrapper, "quant")
        assert hasattr(wrapper, "model")
        assert hasattr(wrapper, "dequant")
        assert isinstance(wrapper.quant, QuantStub)
        assert isinstance(wrapper.dequant, DeQuantStub)
        assert wrapper.model == model


class TestStaticQuantGetAttr:
    def test_getattr_from_dict(self):
        # GIVEN
        #      StaticQuant object is initialized with a model
        #      StaticQuant object has non-module attribute
        wrapper = StaticQuant(nn.Identity())
        expected = "expected"
        wrapper.attr = expected

        # WHEN
        #      attribute is retrieved with __getattr__
        attr = wrapper.__getattr__("attr")

        # THEN
        #      attribute is retrieved correctly
        assert attr == expected

    def test_getattr_from_module_dict(self):
        # GIVEN
        #      StaticQuant object is initialized with a model
        #      The model is stored inside _modules
        model = nn.Identity()
        wrapper = StaticQuant(model)

        # WHEN
        #      module is retrieved with __getattr__
        retrieved_model = wrapper.model

        # THEN
        #      module is retrieved correctly
        assert retrieved_model == model

    def test_getattr_from_wrapped_model(self):
        # GIVEN
        #      StaticQuant object is initialized with a model
        #      The inner model has an attribute
        model = nn.Identity()
        expected = "expected"
        model.attr = expected
        wrapper = StaticQuant(model)

        # WHEN
        #      attribute is retrieved from wrapper
        attr = wrapper.attr

        # THEN
        #      attribute is retrieved correctly from the inner model
        assert attr == expected


class TestStaticQuantForward:
    def test_wrapped_model_returns_single_value(self):
        # GIVEN
        #      StaticQuant object is initialized with a model
        #      The inner model's forward method returns a single value
        #      No quantization is applied to the wrapper
        model = nn.Identity()
        wrapper = StaticQuant(model)
        x = torch.tensor([1.0])
        model.eval()
        wrapper.eval()
        y = model(x)

        # WHEN
        #      the wrapper is applied to an input
        output = wrapper(x)

        # THEN
        #      the output is identical to the output of the inner model
        assert output == y

    def test_wrapped_model_returns_tuple(self):
        # GIVEN
        #      StaticQuant object is initialized with a model
        #      The inner model's forward method returns a tuple
        #      No quantization is applied to the wrapper
        model = nn.Identity()
        model.forward = lambda x: (x, x + 1, torch.tensor([0.0]))
        wrapper = StaticQuant(model)
        x = torch.tensor([1.0])
        model.eval()
        wrapper.eval()
        y = model(x)

        # WHEN
        #      the wrapper is applied to an input
        output = wrapper(x)

        # THEN
        #      the output is identical to the output of the inner model
        assert output == y

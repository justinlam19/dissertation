"""
Wrapper for submodules in order to enable static quantization.
"""

import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub


class StaticQuant(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    # Override __getattr__ so that other code can successfully
    # access attributes of the contained model without erroring.
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]
        else:
            return getattr(self.__dict__["_modules"]["model"], name)
        
    def forward(self, x, *args, **kwargs):
        x = self.quant(x)
        x = self.model(x, *args, **kwargs)
        # ensure that multiple return values are dealt with
        # because DeQuant on its own does not support tuples
        if isinstance(x, tuple):
            return tuple(self.dequant(output) for output in x)
        else:
            return self.dequant(x)

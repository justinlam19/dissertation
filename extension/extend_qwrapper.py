import torch.nn as nn
import torch.nn.functional as F
from torchquant import QWrapper
from torchquant.qmodule import (
    _SUPPORTED_ACTS,
    _SUPPORTED_PATTERNS_BASE,
    Conv2dDynamicSamePadding,
    Conv2dStaticSamePadding,
    _do_padding,
)


class ExtendedQWrapper(QWrapper):
    def parse_sequential_layers(self, layers):
        new_supported_patterns = []
        for pat in _SUPPORTED_PATTERNS_BASE + [[nn.Conv1d]]:
            new_supported_patterns.append(list(pat))
            for act in _SUPPORTED_ACTS:
                new_pat = list(pat)
                new_pat.append(act)
                new_supported_patterns.append(new_pat)
        try:
            types = [type(x) for x in layers]
        except TypeError:  # not iterable
            layers = [layers]
            types = [type(x) for x in layers]

        if not (types in new_supported_patterns):
            raise TypeError(
                f"Provided layers are not supported for fused quantization with QWrapper: {types}."
            )

        self.layer = layers[0]

        try:
            self.bn = next(filter(lambda x: type(x) == nn.BatchNorm2d, layers))
        except StopIteration:
            self.bn = None

        try:
            self.non_linearity = next(
                filter(lambda x: type(x) in _SUPPORTED_ACTS, layers)
            )
        except StopIteration:
            self.non_linearity = None

    def forward(self, x):
        layer = self.layer

        if self.mode.is_weight_observed:
            self.weight_quantizer.pre_observe(layer.weight)

        if self.mode.is_weight_quantized:
            q_weight = self.weight_quantizer(layer.weight)
            self.weight_quantizer.post_observe(q_weight)
        else:
            q_weight = layer.weight

        if isinstance(layer, nn.Conv1d):
            acts = F.conv1d(
                x,
                weight=q_weight,
                bias=layer.bias,
                stride=layer.stride,
                padding=layer.padding,
                groups=layer.groups,
            )
        elif isinstance(layer, nn.Conv2d):
            if isinstance(layer, (Conv2dDynamicSamePadding, Conv2dStaticSamePadding)):
                x = _do_padding(x, layer)

            acts = F.conv2d(
                x,
                weight=q_weight,
                bias=layer.bias,
                stride=layer.stride,
                padding=layer.padding,
                groups=layer.groups,
            )
        elif type(layer) == nn.Linear:
            acts = F.linear(x, q_weight, layer.bias)
        else:
            # We should never get here.
            raise TypeError

        if self.bn is not None:
            # TODO: Implement BatchNorm Folding
            acts = self.bn(acts)

        if self.non_linearity is not None:
            acts = self.non_linearity(acts)

        if self.mode.is_act_observed:
            self.acts_quantizer.pre_observe(acts)

        if self.mode.is_act_quantized:
            acts = self.acts_quantizer(acts)
            self.acts_quantizer.post_observe(acts)

        self.acts = acts

        return acts

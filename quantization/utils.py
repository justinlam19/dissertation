"""
Utility functions for getting nested attributes and attributes inside lists,
specified only by string.
"""

from speechbrain.inference import Pretrained


def get_module(model: Pretrained, module_name: str):
    curr = model.mods
    for attr in module_name.split("."):
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    return curr


def set_module(model: Pretrained, module_name: str, new_module):
    curr = model.mods
    attrs = module_name.split(".")
    for attr in attrs[:-1]:
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    if attrs[-1].isnumeric():
        curr[int(attrs[-1])] = new_module
    else:
        setattr(curr, attrs[-1], new_module)

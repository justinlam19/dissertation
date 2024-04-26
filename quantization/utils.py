from typing import Any
import torch.nn as nn

from speechbrain.inference import Pretrained


def get_attr(obj, attr_name: str):
    curr = obj
    for attr in attr_name.split("."):
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    return curr


def set_attr(obj, attr_name: str, new_attr):
    curr = obj
    attrs = attr_name.split(".")
    for attr in attrs[:-1]:
        if attr.isnumeric():
            curr = curr[int(attr)]
        else:
            curr = getattr(curr, attr)
    if attrs[-1].isnumeric():
        curr[int(attrs[-1])] = new_attr
    else:
        setattr(curr, attrs[-1], new_attr)

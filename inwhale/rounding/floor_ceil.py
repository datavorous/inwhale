import torch

from .base import RoundingStrategy


class FloorRounding(RoundingStrategy):
    """
    Rounds the input tensor down to the nearest integer using the floor function.
    param x: Input tensor to be rounded down.
    return: Tensor rounded down to the nearest integer.
    """
    def round(self, x):
        return torch.floor(x)


class CeilRounding(RoundingStrategy):
    """
    Rounds the input tensor up to the nearest integer using the ceil function.
    param x: Input tensor to be rounded up.
    return: Tensor rounded up to the nearest integer.
    """
    def round(self, x):
        return torch.ceil(x)

import torch

from .base import RoundingStrategy


class FloorRounding(RoundingStrategy):
    def round(self, x):
        return torch.floor(x)


class CeilRounding(RoundingStrategy):
    def round(self, x):
        return torch.ceil(x)

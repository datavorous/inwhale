import torch

from .base import RoundingStrategy


class RoundAwayFromZero(RoundingStrategy):
    def round(self, x):
        return torch.sign(x) * torch.ceil(torch.abs(x))

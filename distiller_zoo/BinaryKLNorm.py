import torch
import torch.nn as nn
from torch.nn import KLDivLoss


class BinaryKLNorm(nn.Module):
    def __init__(self, temperature=2, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.criterion = KLDivLoss(reduction="none")

    def forward(self, student, teacher):
        N, _ = student.shape
        diff_st = torch.sigmoid((student - teacher) / self.temperature)
        diff_st = torch.clamp(diff_st, min=self.eps, max=1-self.eps)
        target = torch.full_like(diff_st, 0.5)
        loss = self.criterion(torch.log(diff_st), target) + self.criterion(torch.log(1 - diff_st), 1 - target)
        return self.temperature ** 2 * loss.sum() / N

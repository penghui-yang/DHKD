import torch
import torch.nn as nn
from torch.nn import KLDivLoss


class BinaryKL(nn.Module):
    def __init__(self, temperature=2, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.criterion = KLDivLoss(reduction="none")

    def forward(self, student, teacher):
        N, _ = student.shape
        student = torch.sigmoid(student / self.temperature)
        teacher = torch.sigmoid(teacher / self.temperature)
        student = torch.clamp(student, min=self.eps, max=1-self.eps)
        teacher = torch.clamp(teacher, min=self.eps, max=1-self.eps)
        loss = self.criterion(torch.log(student), teacher) + self.criterion(torch.log(1 - student), 1 - teacher)
        return self.temperature ** 2 * loss.sum() / N

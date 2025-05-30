import torch.nn as nn


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)

import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, size, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self.std_epsilon = nn.Parameter(torch.tensor(std_epsilon), requires_grad=False)

        self.register_buffer('acc_count', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('num_acc', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('acc_sum', torch.zeros(size, dtype=torch.float32))
        self.register_buffer('acc_sum_squared', torch.zeros(size, dtype=torch.float32))

    def forward(self, data, mean=None, std=None):
        if mean is None:
            mean = self.mean()
        if std is None:
            std = self.std()
        return (data - mean) / std

    def inverse(self, normalized_data, mean=None, std=None):
        if mean is None:
            mean = self.mean()
        if std is None:
            std = self.std()
        return normalized_data * std + mean

    def accumulate(self, data):
        if len(data.shape) == 3:
            b, n, _ = data.shape
            data = data.reshape((b * n, -1))
            data_sum = torch.sum(data, dim=0)
            data_sum_squared = torch.sum(data ** 2, dim=0)
            self.acc_sum += data_sum
            self.acc_sum_squared += data_sum_squared
            self.acc_count += torch.tensor(b * n).to(self.acc_count.device)
            self.num_acc += torch.tensor(1.0).to(self.num_acc.device)
        elif len(data.shape) == 4:
            b, n, t, _ = data.shape
            data = data.reshape((b * n * t, -1))
            data_sum = torch.sum(data, dim=0)
            data_sum_squared = torch.sum(data ** 2, dim=0)
            self.acc_sum += data_sum
            self.acc_sum_squared += data_sum_squared
            self.acc_count += torch.tensor(b * n * t).to(self.acc_count.device)
            self.num_acc += torch.tensor(1.0).to(self.num_acc.device)
        elif len(data.shape) == 2:
            n, _ = data.shape
            data_sum = torch.sum(data, dim=0)
            data_sum_squared = torch.sum(data ** 2, dim=0)
            self.acc_sum += data_sum
            self.acc_sum_squared += data_sum_squared
            self.acc_count += torch.tensor(n).to(self.acc_count.device)
            self.num_acc += torch.tensor(1.0).to(self.num_acc.device)

    def mean(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0).to(self.acc_count.device))
        return self.acc_sum / safe_count

    def std(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0).to(self.acc_count.device))
        std = torch.sqrt(self.acc_sum_squared / safe_count - self.mean() ** 2)
        return torch.maximum(std, self.std_epsilon)

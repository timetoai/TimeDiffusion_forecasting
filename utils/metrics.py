import numpy as np
import torch

def MAE_numpy(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()

def MSE_numpy(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def MAPE_numpy(y_pred, y_true, eps=1e-9):
    return (np.abs(y_pred - y_true) / (eps + np.abs(y_true))).mean()


class MAE_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return (y_pred - y_true).abs().mean()

class MSE_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()


class MAPE_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, eps=1e-9):
        return ((y_pred - y_true).abs() / (eps + y_true.abs())).mean()


class WAPE_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, eps=1e-9):
        return (y_pred - y_true).abs().sum() / (eps + y_true.abs()).sum()


class WMAPE_torch(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self._w = weights

    def forward(self, y_pred, y_true, eps=1e-9):
        return ((y_pred - y_true).abs() * self._w).sum() / (eps + y_true.abs()).sum()
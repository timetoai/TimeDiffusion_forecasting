from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from .dl import QuantGAN_TemporalBlock

def is_high_freq(time_series, threshold=0.5, rolling_parts=200):
    orig_std = time_series.std().values[0]
    ma_ts = time_series.rolling(len(time_series) // rolling_parts).mean()
    ma_std = ma_ts.std().values[0]
    return abs(ma_std - orig_std) / orig_std > threshold

def ma(time_series, rolling_parts=200, window=None):
    if window is None:
        window = max(len(time_series) // rolling_parts, 2)
    ts1 = time_series.rolling(window, closed="left").mean()
    ts2 = time_series[:: - 1].rolling(window).mean()[:: - 1]
    ts1[ts1.isna()] = ts2[ts1.isna()]
    ts2[ts2.isna()] = ts1[ts2.isna()]
    ats = (ts1 + ts2) / 2
    return ats


class TimeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.ModuleList([QuantGAN_TemporalBlock(1, 128, kernel_size=1, stride=1, dilation=1, padding=0, dropout=0.25),
                                 *[QuantGAN_TemporalBlock(128, 128, kernel_size=2, stride=1, dilation=i, padding=i, dropout=0.0)
                                        for i in [2 ** i for i in range(14)]]])
        self.last = nn.Conv1d(128, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x

def train_forecast(seq: np.array, horizon: int, device, epochs=20, seed=0, lr=4e-4, batch_size=32, steps_per_epoch=32, pbar=False):
    """
    Trains TimeDiffusion model on sequence and forecast for horizon length
    """
    tmean = seq.mean()
    tstd = seq.std()
    train = (seq - tmean) / tstd
    train_tensor = torch.from_numpy(train).float().to(device)

    torch.random.manual_seed(seed)
    model = TimeDiffusion().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for epoch in (tqdm(range(1, epochs + 1)) if pbar else range(1, epochs + 1)):
        model.train()
        X = train_tensor.repeat(batch_size, 1).unsqueeze(1)
        noise = torch.row_stack([torch.rand(1, *X.shape[1:]) for _ in range(X.shape[0])]).to(device)
        noise_level = torch.rand(X.shape).to(device)
        noise *= noise_level

        for step in range(steps_per_epoch):
            optim.zero_grad()
            y_hat = model(X + noise)
            loss = (y_hat - noise).abs().mean()
            loss.backward()
            optim.step()
            with torch.no_grad():
                noise -= y_hat
            losses.append(loss.item())

    model.eval()
    result = []
    with torch.no_grad():
        generated = torch.rand(1, 1, len(seq) + horizon).to(device)
        for step in range(1, steps_per_epoch + 1):
            pred_noise = model(generated)
            generated -= pred_noise
            result.append(generated.detach().cpu().numpy().squeeze() * tstd + tmean)
    result = np.row_stack(result)

    return {"model": model, "forecast": result, "losses": losses}

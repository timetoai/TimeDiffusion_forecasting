from tqdm import tqdm

import numpy as np

import torch
from torch import nn

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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : - self.chomp_size].contiguous()


class QuantGAN_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(QuantGAN_TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        if dropout != 0:
            self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        if dropout != 0:
            self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            if dropout == 0:
                self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
            else:
                self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        else:
            if dropout == 0:
                self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2)
            else:
                self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.5)
        self.conv2.weight.data.normal_(0, 0.5)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)


class TimeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.ModuleList([QuantGAN_TemporalBlock(1, 128, kernel_size=1, stride=1, dilation=1, padding=0, dropout=0.25),
                                 *[QuantGAN_TemporalBlock(128, 128, kernel_size=2, stride=1, dilation=i, padding=i, dropout=0.0)
                                        for i in [2 ** i for i in range(14)]]])
        self.last = nn.Conv1d(128, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_acc = None
        for layer in self.tcn:
            skip, x = layer(x)
            if skip_acc is None:
                skip_acc = skip
            else:
                skip_acc += skip
        x = self.last(x + skip_acc)
        return x

def train_forecast(seq: np.array, horizon: int, device, epochs=20, seed=0, lr=4e-4, batch_size=32, steps_per_epoch=32, pbar=False):
    """
    Trains TimeDiffusion model on sequence and forecast for horizon length
    """
    tmean = seq.mean()
    tstd = seq.std()
    train = (seq - tmean) / tstd
    train_tensor = torch.from_numpy(train).to(device=device, dtype=torch.float32)

    torch.random.manual_seed(seed)
    model = TimeDiffusion().to(device=device, dtype=torch.float32)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for epoch in (tqdm(range(1, epochs + 1)) if pbar else range(1, epochs + 1)):
        model.train()
        X = train_tensor.repeat(batch_size, 1).unsqueeze(1)
        noise = torch.row_stack([torch.rand(1, *X.shape[1:]) for _ in range(X.shape[0])]).to(device=device, dtype=torch.float32)
        noise_level = torch.rand(X.shape).to(device=device, dtype=torch.float32)
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
        generated = torch.rand(1, 1, len(seq) + horizon).to(device=device, dtype=torch.float32)
        for step in range(1, steps_per_epoch + 1):
            pred_noise = model(generated)
            generated -= pred_noise
            result.append(generated.detach().cpu().numpy().squeeze() * tstd + tmean)
    result = np.row_stack(result)

    return {"model": model, "forecast": result, "losses": losses}

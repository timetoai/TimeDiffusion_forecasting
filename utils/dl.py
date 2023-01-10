import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything

import torch
from torch import nn

class Model:
    """
    Wrapper for models, containing train / eval functions
    """
    def __init__(self, seed=0, device='cpu'):
        seed_everything(seed)
        self.device = device

        self.model = self.optim = \
            self.criterion = self._last_losses = None

    def set_model(self, model_init, **model_params):
        self.model = model_init(**model_params).to(self.device)

    def get_number_model_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.model.parameters()])

    def set_optim(self, optim_init, **optim_params):
        self.optim = optim_init(**optim_params)

    def set_criterion(self, criterion_init, **criterion_params):
        self.criterion = criterion_init(**criterion_params)

    def train_step(self, X, y, do_back=True):
        self.model.train()
        if do_back:
            self.model.zero_grad()
        y_preds = self.model(X)
        loss = self.criterion(y_preds, y)
        loss.backward()
        if do_back:
            self.optim.step()
        return loss.item()

    def train(self, train_dataloader, val_dataloader=None, epochs=1, print_info=True, agg_loss="mean"):
        losses = {'train': [], 'val': []}

        for epoch_num in range(1, epochs + 1):
            losses['train'].append(0)
            for X, y in train_dataloader:
                loss = self.train_step(X, y, do_back=True)
                losses['train'][- 1] += loss
            
            
            if agg_loss == "mean":
                loss /= len(train_dataloader)
            elif agg_loss == "sum":
                pass
            else:
                raise
            
            if val_dataloader:
                losses['val'].append(self.eval(val_dataloader, agg_loss=agg_loss))
        
            if print_info:
                print(f"Epoch #{epoch_num}: train loss {losses['train'][- 1]: 0.6f}, val loss {losses['val'][- 1] if losses['val'] else 0: 0.6f}")

        self._last_losses = {key: value for key, value in losses.items() if value}
        return self._last_losses

    def get_min_losses(self, losses=None):
        if losses is None:
            losses = self._last_losses
        return {key: (np.argmin(value), min(value)) for key, value in self._last_losses.items()}

    def get_last_losses(self, losses=None):
        if losses is None:
            losses = self._last_losses
        return {key: value[- 1] for key, value in self._last_losses.items()}

    def plot_losses(self, losses=None):
        if losses is None:
            losses = self._last_losses
        return pd.DataFrame(losses).plot()

    def inference(self, X):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
        return preds

    def eval(self, dataloader, return_preds=False, agg_loss="mean"):
        self.model.eval()
        loss = 0
        if return_preds:
            preds = []
        with torch.no_grad():
            for X, y in dataloader:
                y_preds = self.model(X)
                if return_preds:
                    preds.append(y_preds)
                loss += self.criterion(y_preds, y).item()
        if agg_loss == "mean":
            loss /= len(dataloader)
        elif agg_loss == "sum":
            pass
        else:
            raise
        if return_preds:
            return loss, preds
        return loss


class RNNModel(Model):
    """
    Wrapper for Simple RNN models, containing train / eval functions
    """
    def train_step(self, X, y, do_back=True, hs=None):
        self.model.train()
        if do_back:
            self.model.zero_grad()
        y_preds, hs = self.model(X, hs)
        loss = self.criterion(y_preds, y)
        loss.backward()
        if do_back:
            self.optim.step()
        return loss.item(), hs

    def train(self, train_dataloader, val_dataloader=None, epochs=1, print_info=True, keep_hs=False, agg_loss="mean"):
        losses = {'train': [], 'val': []}

        for epoch_num in range(1, epochs + 1):
            hs = None
            losses['train'].append(0)
            for X, y in train_dataloader:
                loss, hs = self.train_step(X, y, do_back=True, hs=hs)
                losses['train'][- 1] += loss
                if not keep_hs:
                    hs = None
                else:
                    hs = torch.tensor(hs) if isinstance(hs, torch.Tensor) else [torch.tensor(x) for x in hs]
            
            if agg_loss == "mean":
                loss /= len(train_dataloader)
            elif agg_loss == "sum":
                pass
            else:
                raise

            if val_dataloader:
                losses['val'].append(self.eval(val_dataloader, keep_hs=keep_hs, agg_loss=agg_loss))
        
            if print_info:
                print(f"Epoch #{epoch_num}: train loss {losses['train'][- 1]: 0.6f}, val loss {losses['val'][- 1] if losses['val'] else 0: 0.6f}")

        self._last_losses = {key: value for key, value in losses.items() if value}
        return self._last_losses

    def inference(self, X, hs=None):
        self.model.eval()
        with torch.no_grad():
            preds, _ = self.model(X, hs)
        return preds

    def eval(self, dataloader, return_preds=False, keep_hs=False, hs=None, agg_loss="mean"):
        self.model.eval()
        loss = 0
        if return_preds:
            preds = []
        with torch.no_grad():
            for X, y in dataloader:
                y_preds, hs = self.model(X, hs)
                if return_preds:
                    preds.append(y_preds)
                if not keep_hs:
                    hs = None
                loss += self.criterion(y_preds, y).item()
        if agg_loss == "mean":
            loss /= len(dataloader)
        elif agg_loss == "sum":
            pass
        else:
            raise
        if return_preds:
            return loss, preds
        return loss


class SimpleGRU(nn.Module):
    """
    Neural network consisting of GRU and Linear layers. GRU output passed through network
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, seq_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size * seq_len, output_size)
    
    def forward(self, x, hs=None):
        out, hs = self.gru(x) if hs is None else self.gru(x, hs)
        # out = self.fc(out.view(out.shape[0], - 1))
        out = self.fc(out.reshape((out.shape[0], - 1)))
        return out, hs


class SimpleHiddenGRU(nn.Module):
    """
    Neural network consisting of GRU and Linear layers. GRU hidden state passed through network
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, seq_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(num_layers * hidden_size, output_size)
    
    def forward(self, x, hs=None):
        out, hs = self.gru(x) if hs is None else self.gru(x, hs)
        hs_forward = hs.permute(1, 0, 2).contiguous().view(hs.shape[1], - 1)
        out = self.fc(hs_forward)
        return out, hs


class SimpleLSTM(nn.Module):
    """
    Neural network consisting of LSTM and Linear layers. LSTM output passed through network
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size * seq_len, output_size)
    
    def forward(self, x, hs=None):
        out, (hs, cs) = self.lstm(x) if hs is None else self.lstm(x, hs)
        # out = self.fc(out.view(out.shape[0], - 1))
        out = self.fc(out.reshape((out.shape[0], - 1)))
        return out, (hs, cs)


class SimpleHiddenLSTM(nn.Module):
    """
    Neural network consisting of LSTM and Linear layers. LSTM hidden state passed through network
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(num_layers * hidden_size, output_size)
    
    def forward(self, x, hs=None):
        out, (hs, cs) = self.lstm(x) if hs is None else self.lstm(x, hs)
        hs_forward = hs.permute(1, 0, 2).contiguous().view(hs.shape[1], - 1)
        out = self.fc(hs_forward)
        return out, (hs, cs)


class Chomp1d(nn.Module):
    """
    Chomp1d pytorch operation, adapted from https://github.com/locuslab/TCN
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : - self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal Block pytorch operation, adapted from https://github.com/locuslab/TCN
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Pure (without dense layers) Temporal Convolutional Network, adapted from https://github.com/locuslab/TCN
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """
    Temporal Convolutional Network, adapted from https://github.com/locuslab/TCN
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, activation=None):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[- 1], output_size)
        self.activation = activation

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        # output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.tcn(x).squeeze()
        output = self.linear(output)
        return output if self.activation is None else self.activation(output)


class QuantGAN_TemporalBlock(nn.Module):
    """Creates a temporal block.
    Args:
        n_inputs (int): number of inputs.
        n_outputs (int): size of fully connected layers.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        padding (int): padding
        dropout (float): dropout rate
    Returns:
        tuple of output layers
    Adapted from https://github.com/JamesSullivan/temporalCN
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(QuantGAN_TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
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


class QuantGAN_Generator(nn.Module):
    """Generator: 3 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
       Adapted from https://github.com/JamesSullivan/temporalCN
    """ 
    def __init__(self):
        super(QuantGAN_Generator, self).__init__()
        self.tcn = nn.ModuleList([QuantGAN_TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[QuantGAN_TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x


class QuantGAN_Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
       Adapted from https://github.com/JamesSullivan/temporalCN
    """ 
    def __init__(self, seq_len, conv_dropout=0.05):
        super(QuantGAN_Discriminator, self).__init__()
        self.tcn = nn.ModuleList([QuantGAN_TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[QuantGAN_TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()
import numpy as np
import torch

from .dl import TCN, Model
from .metrics import MAE_torch as MAE
from .data import split_data, create_ts_dl

def tcn_train_forecast(ts, val_size, test_size, horizon, lags, stride=1, epochs=40, batch_size=256):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    verbose = False
    drop_last = False

    train_dl, _, test_dl, X_scaler, y_scaler = create_ts_dl(ts.reshape(- 1, 1), ts.flatten(), lags=lags, horizon=horizon, stride=stride,\
                                                    batch_size=batch_size, device=device, data_preprocess=("normalize",),\
                                                    val_size=val_size, test_size=test_size, drop_last=drop_last)

    model_params = {'num_channels': [128] * 4, 'kernel_size': 2, 'dropout': 0.25, 'output_size': horizon, 'input_size': lags}

    model = Model(seed=0, device=device)
    model.set_model(TCN, **model_params)
    
    optim_params = {'params': model.model.parameters(), 'lr': 4e-4}
    model.set_optim(torch.optim.AdamW, **optim_params)
    model.set_criterion(MAE)

    model.train(train_dl, epochs=epochs, print_info=verbose, agg_loss="mean")
    preds = []
    y_true = []
    for X, y in test_dl:
        preds.append(model.inference(X).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())
    preds = np.row_stack(preds)
    y_true = np.row_stack(y_true)

    return {"y": y_true, "preds": preds}
    

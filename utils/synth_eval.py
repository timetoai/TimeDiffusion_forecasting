from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import kl_div as scipy_kl_div
from scipy.stats import kstest

import torch

from .dl import SimpleLSTM, TCN, Model, RNNModel
from .metrics import MAE
from .data import get_hsm_dataset, get_solar_energy_dataset, get_fuel_prices_dataset, get_passengers_dataset,\
     log_returns, get_dataset_iterator, create_ts_dl


def min_max_norm(arr):
    arr_max, arr_min = arr.max(), arr.min()
    if arr_max == arr_min:
        return np.ones_like(arr) * arr_min
    return (arr - arr_min) / (arr_max - arr_min)

def kl_div(x, y):
    x, y = map(min_max_norm, (x, y))
    return scipy_kl_div(x, y)
    
def js_div(x, y):
    x, y = (map(min_max_norm, (x, y)))
    x_y = (x + y) / 2
    return (scipy_kl_div(x, x_y) + scipy_kl_div(y, x_y)) / 2

def eval_sim(dataset_names, dataset_paths, model_name, save=True, results_dir=None):
    """"
    Evaluates similarity of synthetic time series on predifined datasets of selected model
    """
    ret = defaultdict(dict)
    for ds_ind, (dataset_path, dataset_name) in enumerate(zip(dataset_paths, dataset_names)):
        print(f"processing {dataset_name} dataset")
        synthetic_path = dataset_path / f"synthetic/{model_name}/"
        results = {"js_div": [], "kstest_pval": []}
        if dataset_name == "hsm":
            ts_iterator = get_hsm_dataset(dataset_path, selected_files=f"{dataset_path}/selected100.csv")
        elif dataset_name == "se":
            ts_iterator = get_solar_energy_dataset(dataset_path)
        elif dataset_name == "fp":
            ts_iterator = get_fuel_prices_dataset(dataset_path)
        else:
            ts_iterator = get_passengers_dataset(dataset_path)

        for ts_index, time_series in tqdm(enumerate(ts_iterator)):
            
            # train_ts = log_returns(time_series[:10_000]).values.flatten()
            train_ts = time_series.values.flatten()
            if "RealNVP" in model_name or "Flow" in model_name:
                train_ts = train_ts[:(len(train_ts) // 4 * 4 + 1 if len(train_ts) % 4 > 0 else len(train_ts) - 3)]
            # train_ts = min_max_norm(train_ts)
            
            synth_tss = np.load(synthetic_path / f"selected{ts_index}.npy").squeeze()
            if model_name == "TimeDiffusion":
                synth_tss = synth_tss[- 2:]
            if len(synth_tss) > 0:
                js_div_res = []
                p_val = []
                for synth_ts in synth_tss:
                    # synth_ts = min_max_norm(synth_ts)
                    if len(synth_ts) < len(train_ts):
                        for i in range(0, len(train_ts) // len(synth_ts) * len(synth_ts), len(synth_ts)):
                            res = js_div(synth_ts, train_ts[i: i + len(synth_ts)])
                            js_div_res.append(res.mean())
                            # p_val.append(kstest(synth_ts, train_ts[i: i + len(synth_ts)])[1])
                            p_val.append(kstest(min_max_norm(synth_ts), min_max_norm(train_ts[i: i + len(synth_ts)]))[1])
                    else:
                        res = js_div(synth_ts, train_ts[:len(synth_ts)])
                        js_div_res.append(res.mean())
                        # p_val.append(kstest(synth_ts[:len(train_ts)], train_ts)[1])
                        p_val.append(kstest(min_max_norm(synth_ts[:len(train_ts)]), min_max_norm(train_ts))[1])
                results["js_div"].append(np.mean(js_div_res))
                results["kstest_pval"].append(np.mean(p_val))
            else:
                results["js_div"].append(1)
                results["kstest_pval"].append(1)
            
        if save:
            pd.DataFrame(results).to_csv(results_dir / f"synth_{dataset_name}_sim_{model_name}.csv", index=False)
        for key in results:
            ret[dataset_name][key] = np.mean(results[key])
    return ret

def get_model_autoreg_init_params(model_name="LSTM"):
    lags = 32
    horizon = 8
    features = 1
    if model_name == "LSTM":
        model_params = {'input_size': features, 'hidden_size': 256, 'num_layers': 2, 'dropout': 0.1, 'output_size': horizon, 'seq_len': lags}
    elif model_name == "TCN":
        model_params = {'num_channels': [128] * 4, 'kernel_size': 2, 'dropout': 0.25, 'output_size': horizon, 'input_size': lags}

    return model_params

def eval_autoreg_model_real(dataset_names, dataset_paths, model_name="LSTM", results_dir=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lags = 32
    horizon = 8
    stride = 1
    batch_size = 256
    test_size = 0.3
    verbose = False
    drop_last = False

    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        if dataset_name == "se":
            epochs = 5
        else:
            epochs = 40
        
        results = []
        for time_series in tqdm(get_dataset_iterator(dataset_name, dataset_path)):
            target_col = time_series.columns[0]
            train_dl, _, test_dl, X_scaler, y_scaler = create_ts_dl(time_series[[target_col]], time_series[target_col], lags=lags, horizon=horizon, stride=stride,\
                                                batch_size=batch_size, device=device, data_preprocess=("normalize",),\
                                                val_size=0, test_size=test_size, drop_last=drop_last)

            model_params = get_model_autoreg_init_params(model_name)
            if model_name == "LSTM":
                model = RNNModel(seed=0, device=device)
                model.set_model(SimpleLSTM, **model_params)
            elif model_name == "TCN":
                model = Model(seed=0, device=device)
                model.set_model(TCN, **model_params)
            optim_params = {'params': model.model.parameters(), 'lr': 4e-4}
            model.set_optim(torch.optim.AdamW, **optim_params)
            model.set_criterion(MAE)

            model.train(train_dl, epochs=epochs, print_info=verbose, agg_loss="mean")

            model.train(train_dl, epochs=epochs, print_info=verbose, agg_loss="mean")
            results.append(model.eval(test_dl, agg_loss="mean"))

            del model, train_dl, test_dl
            torch.cuda.empty_cache()
    
        pd.DataFrame(results, columns=["test"]).to_csv(results_dir / f"real_{dataset_name}_{model_name}.csv", index=False)

def eval_autoreg_model_synth(dataset_names, dataset_paths, synth_model_name, model_name="LSTM", results_dir=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lags = 32
    horizon = 8
    stride = 1
    batch_size = 256
    val_size = 0.0
    test_size = 0.3
    verbose = False
    drop_last = False
    ds_lens = {"hsm": 100, "se": 10, "fp": 8, "ap": 50}

    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        if dataset_name == "se":
            epochs = 5
        else:
            epochs = 40
        synth_path = dataset_path / "synthetic" / synth_model_name
        results = []
        for ts_index in tqdm(range(ds_lens[dataset_name])):
            synth_time_series = np.load(synth_path / f"selected{ts_index}.npy")
            results.append(0)
            num_synth_samples = min(10 if synth_model_name in ("TTS_GAN", "QuantGAN") else 4, synth_time_series.shape[0])
            if synth_model_name == "TimeDiffusion": num_synth_samples = 2
            synth_range = range(len(synth_time_series) - num_synth_samples, len(synth_time_series))
            for i in synth_range:
                train_dl, _, test_dl, X_scaler, y_scaler = create_ts_dl(synth_time_series[i].reshape(- 1, 1), synth_time_series[i].flatten(), lags=lags, horizon=horizon, stride=stride,\
                                                    batch_size=batch_size, device=device, data_preprocess=("normalize",),\
                                                    val_size=val_size, test_size=test_size, drop_last=drop_last)

                model_params = get_model_autoreg_init_params(model_name)
                if model_name == "LSTM":
                    model = RNNModel(seed=0, device=device)
                    model.set_model(SimpleLSTM, **model_params)
                elif model_name == "TCN":
                    model = Model(seed=0, device=device)
                    model.set_model(TCN, **model_params)
                optim_params = {'params': model.model.parameters(), 'lr': 4e-4}
                model.set_optim(torch.optim.AdamW, **optim_params)
                model.set_criterion(MAE)

                model.train(train_dl, epochs=epochs, print_info=verbose, agg_loss="mean")
                results[- 1] += model.eval(test_dl, agg_loss="mean")

                del model, train_dl, test_dl
                torch.cuda.empty_cache()
            results[- 1] /= num_synth_samples
            del synth_time_series

        pd.DataFrame(results, columns=["test"]).to_csv(results_dir / f"synth_{synth_model_name}_{dataset_name}_{model_name}.csv", index=False)

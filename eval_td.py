import argparse

def main(args):
    from pathlib import Path
    from tqdm import tqdm

    import numpy as np
    import pandas as pd
    import torch

    from utils.data import get_exchange_rate_dataset, get_ett_dataset, get_etl_dataset,\
         build_ts_X_y, DimUniversalStandardScaler, DimUniversalMinMaxScaler,\
         pickle_save, pickle_load
    from utils.metrics import MAE_numpy as MAE, MSE_numpy as MSE
    from utils.timediffusion import train_forecast

    results_dir, data_dir = map(Path, (args.results, args.data))
    dataset_name, epochs, steps_per_epoch = args.dataset, args.epochs, args.steps
    test_subsamples_limit = args.test_subsamples_limit
    test_ts_limit = args.test_timeseries_limit
    if not results_dir.exists(): results_dir.mkdir()
    results_dir /= dataset_name
    if not results_dir.exists(): results_dir.mkdir()
    if not data_dir.exists(): raise "Data directory not found"
    if dataset_name not in ("Exchange", "ETT", "ETL"): raise "Specify proper dataset name"

    get_ds_iterator = lambda: {"Exchange": get_exchange_rate_dataset, "ETT": get_ett_dataset,
                    "ETL": get_etl_dataset}[dataset_name](data_dir)
    horizons = [96, 192, 336, 720]
    get_lags = lambda h: h * 5 if h < 500 else h * 2
    train_size = 0.7
    test_size = 0.2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    results = {"mae": [], "mse": []}
    pbar = tqdm(horizons)
    for horizon in pbar:
        results["mae"].append(0)
        results["mse"].append(0)
        for ts_ind, ts in enumerate(get_ds_iterator()):
            ts = ts[- get_lags(horizon) - int(len(ts) * test_size):]
            Xs, ys = build_ts_X_y(X=ts, y=ts, lags=get_lags(horizon), horizon=horizon, stride=horizon)
            # print(horizon, ts_count, len(ts), len(ys))
            if len(Xs) > test_subsamples_limit:
                Xs = Xs[- test_subsamples_limit:]
                ys = ys[- test_subsamples_limit:]
            preds = []
            for i, (X, y) in enumerate(zip(Xs, ys)):
                pbar.set_description(f"Processing {dataset_name} horizon {horizon} time series {ts_ind + 1} {i}/{len(ys)}")
                res = train_forecast(X, horizon, device, epochs=epochs, steps_per_epoch=steps_per_epoch)
                y_pred = res["forecast"][- 1][- horizon:]
                preds.append(y_pred)
            preds = np.row_stack(preds)
            results["mae"][- 1] += MAE(preds, ys)
            results["mse"][- 1] += MSE(preds, ys)
            pickle_save(dict(x=Xs, y=ys, preds=preds), results_dir / f"ts{ts_ind}_horizon{horizon}.pkl")
            if ts_ind + 1 == test_ts_limit:
                break
        results["mae"][- 1] /= ts_ind + 1
        results["mse"][- 1] /= ts_ind + 1
        print(results)
    results = pd.DataFrame(results)
    results["horizon"] = horizons
    results.to_csv(results_dir / f"TD_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, action="store", required=True, help="Exchange|ETT|ETL")
    parser.add_argument("--data", type=str, action="store", required=True, help="Filepath or directory with dataset (for ETT dataset)")
    parser.add_argument("--results", type=str, action="store", default="results" , help="Directory with results files")
    parser.add_argument("--epochs", type=int, action="store", default=10, help="Epochs to train TimeDiffusion model")
    parser.add_argument("--steps", type=int, action="store", default=32, help="Steps per epoch")
    parser.add_argument("--test_subsamples_limit", type=int, action="store", default=2,
                     help="Max test subsamples for each time series (for computation speed up)")
    parser.add_argument("--test_timeseries_limit", type=int, action="store", default=8,
                     help="Max test time series from the dataset (for computation speed up)")
    args = parser.parse_args()
    main(args)


import argparse

def main(args):
    from pathlib import Path
    from tqdm import tqdm

    import pandas as pd
    import torch

    from utils.data import get_exchange_rate_dataset, get_ett_dataset, get_etl_dataset, build_ts_X_y
    from utils.metrics import MAE_numpy as MAE, MSE_numpy as MSE
    from utils.timediffusion import train_forecast

    results_dir, data_dir = map(Path, (args.results, args.data))
    dataset_name, epochs, steps_per_epoch = args.dataset, args.epochs, args.steps
    if not results_dir.exists(): results_dir.mkdir()
    if not data_dir.exists(): raise "Data directory not found"
    if dataset_name not in ("Exchange", "ETT", "ETL"): raise "Specify proper dataset name"

    get_ds_iterator = lambda: {"Exchange": get_exchange_rate_dataset, "ETT": get_ett_dataset,
                    "ETL": get_etl_dataset}[dataset_name](data_dir)
    horizons = [96, 192, 336, 720]
    get_lags = lambda h: h * 5 if h < 500 else h * 2
    test_size = 0.2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    results = {"mae": [], "mse": []}
    pbar = tqdm(horizons)
    for horizon in pbar:
        results["mae"].append(0)
        results["mse"].append(0)
        ts_count = 0
        for ts in get_ds_iterator():
            ts_count += 1
            mse = mae = 0
            ts = ts[- get_lags(horizon) - int(len(ts) * test_size):]
            Xs, ys = build_ts_X_y(X=ts, y=ts, lags=get_lags(horizon), horizon=horizon, stride=horizon)
            # print(horizon, ts_count, len(ts), len(ys))
            for i, (X, y) in enumerate(zip(Xs, ys)):
                pbar.set_description(f"Processing {dataset_name} horizon {horizon} time series {ts_count} {i}/{len(ys)}")
                res = train_forecast(X, horizon, device, epochs=epochs, steps_per_epoch=steps_per_epoch)
                y_pred = res["forecast"][- 1][- horizon:]
                mae += MAE(y_pred, y)
                mse += MSE(y_pred, y)
            mae /= len(ys)
            mse /= len(ys)
            results["mae"][- 1] += mae
            results["mse"][- 1] += mse
        if ts_count:
            results["mae"][- 1] /= ts_count
            results["mse"][- 1] /= ts_count
        print(results)
    results = pd.DataFrame(results)
    results["horizon"] = horizons
    results.to_csv(results_dir / f"TD_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, action="store", required=True, help="Exchange|ETT|ETL")
    parser.add_argument("--data", type=str, action="store", required=True, help="Filepath or directory with dataset (for ETT dataset)")
    parser.add_argument("--results", type=str, action="store", default="results" , help="Directory with results files")
    parser.add_argument("--epochs", type=int, action="store", default=4, help="Epochs to train TimeDiffusion model")
    parser.add_argument("--steps", type=int, action="store", default=16, help="Steps per epoch")
    args = parser.parse_args()
    main(args)


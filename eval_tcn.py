import argparse

def main(args):
    from pathlib import Path
    from tqdm import tqdm

    import numpy as np
    import pandas as pd

    from utils.data import get_exchange_rate_dataset, get_ett_dataset, get_etl_dataset,\
         DimUniversalStandardScaler, pickle_save
    from utils.TCN import tcn_train_forecast

    results_dir, data_dir = map(Path, (args.results, args.data))
    dataset_name, epochs = args.dataset, args.epochs
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
    train_part = 0.7
    val_part = 0.8

    results = {"mae": [], "mse": []}
    pbar = tqdm(horizons)
    for horizon in pbar:
        results["mae"].append(0)
        results["mse"].append(0)
        for ts_ind, ts in enumerate(get_ds_iterator()):
            res = tcn_train_forecast(ts, val_size=0.1, test_size=0.2, horizon=horizon, lags=get_lags(horizon), epochs=epochs)
            
            mae = np.mean(np.abs((res["preds"] - res["y"])))
            mse = np.mean((res["preds"] - res["y"]) ** 2)
            results["mae"][- 1] += mae.item()
            results["mse"][- 1] += mse.item()
            pickle_save(dict(x=None, y=res["y"], preds=res["preds"]),\
                         results_dir / f"ts{ts_ind}_horizon{horizon}.pkl")
            if ts_ind + 1 == test_ts_limit:
                break
        results["mae"][- 1] /= ts_ind + 1
        results["mse"][- 1] /= ts_ind + 1
        print(results)
    results = pd.DataFrame(results)
    results["horizon"] = horizons
    results.to_csv(results_dir / f"tcn_{dataset_name}_norm.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, action="store", required=True, help="Exchange|ETT|ETL")
    parser.add_argument("--data", type=str, action="store", required=True, help="Filepath or directory with dataset (for ETT dataset)")
    parser.add_argument("--results", type=str, action="store", default="results" , help="Directory with results files")
    parser.add_argument("--epochs", type=int, action="store", default=5, help="Epochs to train model")
    parser.add_argument("--test_timeseries_limit", type=int, action="store", default=8,
                     help="Max test time series from the dataset (for computation speed up)")
    args = parser.parse_args()
    main(args)


from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import torch

def get_hsm_dataset(dataset_path, selected_files=None):
    """
    Creates generator for time series from `huge stock market dataset`
    Dataset URL: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
    """
    dataset_path = Path(dataset_path)
    if selected_files is None:
        for subfolder in dataset_path.iterdir():
            if not subfolder.is_dir(): continue
            for file in subfolder.iterdir():
                # yield pd.read_csv(file, index_col="Date", parse_dates=["Date"])
                yield pd.read_csv(file, usecols=["Close"])  # fastest variant
    else:
        selected_files = pd.read_csv(Path(selected_files)).filename.values
        for filename in selected_files:
            file = list(dataset_path.glob(f"*/{filename}"))[0]
            yield pd.read_csv(file, usecols=["Close"])

def get_solar_energy_dataset(dataset_path, max_results=10):
    dataset_path = Path(dataset_path) / "al-pv-2006"
    for path in dataset_path.glob("*Actual*"):
        yield pd.read_csv(path, usecols=["Power(MW)"]).iloc[:10_000]
        max_results -= 1
        if max_results == 0:
            break

def get_fuel_prices_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path / "weekly_fuel_prices_all_data_from_2005_to_20210823.csv")
    missing = set((4, 7))
    for i in range(1, 9):
        if i not in missing:
            yield df[df.product_id == i].sort_values("survey_date")[["price"]]

    df = pd.read_csv(dataset_path / "Weekly Fuel Prices.csv").sort_values("Date")
    for col in ("Petrol (USD)", "Diesel (USD)"):
        yield df[[col]]

def get_passengers_dataset(dataset_path, max_results=50):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path / "US Monthly Air Passengers.csv")
    with open(dataset_path / "cities", "rb") as f:
        cities = pickle.load(f)
        
    for city in cities[:max_results]:
        yield df[df.ORIGIN_CITY_NAME == city].\
            groupby(["YEAR", "MONTH"], as_index=False).agg(passengers=("Sum_PASSENGERS", "sum")).\
                sort_values(["YEAR", "MONTH"])[["passengers"]]

def get_exchange_rate_dataset(filepath):
    df = pd.read_csv(filepath)
    for col in df.columns:
        yield df[col].values.flatten()

def get_ett_dataset(dataset_path):
    # for filename in ("ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"):
    for filename in ("ETTm2.csv",):
        df = pd.read_csv(dataset_path / filename)
        for col in ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"):
            yield df[col].values.flatten()

def get_etl_dataset(filepath):
    df = pd.read_csv(filepath)
    for col in (f"MT_{i:0>3}" for i in range(1, 371)):
        ts = np.trim_zeros(df[col].values.flatten())
        yield ts

def get_dataset_iterator(dataset_name, dataset_path):
    if dataset_name == "hsm":
        ts_iterator = get_hsm_dataset(dataset_path, selected_files=f"{dataset_path}/selected100.csv")
    elif dataset_name == "se":
        ts_iterator = get_solar_energy_dataset(dataset_path)
    elif dataset_name == "fp":
        ts_iterator = get_fuel_prices_dataset(dataset_path)
    else:
        ts_iterator = get_passengers_dataset(dataset_path)
    return ts_iterator

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def log_returns(series: pd.Series) -> pd.Series:
    """
    Takes pandas.Series as input and returns it in `log returns` format
    """
    return np.log(series / series.shift(1)).fillna(0)

def inverse_log_returns(time_series, start_value: int):
    ts = np.exp(time_series)
    ts[0] = start_value
    return ts.cumprod()

def build_ts_X_y(X, y, lags=1, horizon=1, stride=1):
    """
    Builds arrays for model training for time series data
    """
    X = np.concatenate([[X[i - lags: i]] for i in range(lags, len(y) - horizon + 1, stride)], axis=0)
    y = np.row_stack([y[i: i + horizon] for i in range(lags, len(y) - horizon + 1, stride)])
    return X, y

def split_data(*arrs, val_size=0.15, test_size=0.15, rate=1):
    """
    Splits data into train / val / test parts taking into account data rate
    """
    val_len = round(len(arrs[0] / rate) * val_size) * rate
    test_len = round(len(arrs[0] / rate) * test_size) * rate

    arrs = [(arr[: len(arr) - val_len - test_len], arr[len(arr) - val_len - test_len: len(arr) - test_len],\
                                arr[len(arr) - test_len:]) for arr in arrs]
    return arrs

def normalize(train, *others):
    """
    Normalizes samples based on train distribution using sklearn StandardScaler
    returns: train, *others, scaler
    """
    scaler = DimUniversalStandardScaler()
    train = scaler.fit_transform(train)
    return train, *[scaler.transform(x) if x.size > 0 else x for x in others], scaler

def create_ts(X, y, lags, horizon, stride, val_size, test_size, data_preprocess=("log_returns", "normalize"), rate=1, scaler=None):
    """
    Full pipeline of building train / val / test parts
    """
    if "log_returns" in data_preprocess:
        X = log_returns(X)
        y = log_returns(y)
    X, y = build_ts_X_y(X, y, lags=lags, horizon=horizon, stride=stride)

    (X_train, X_val, X_test), (y_train, y_val, y_test) = split_data(X, y.reshape((len(X), - 1)), val_size=val_size, test_size=test_size, rate=rate)
    if "normalize" in data_preprocess:
        if scaler is None:
            X_train, X_val, X_test, std_scaler_X = normalize(X_train, X_val, X_test)
            y_train, y_val, y_test, std_scaler_y = normalize(y_train, y_val, y_test)
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = map(scaler.transform, (X_train, X_val, X_test, y_train, y_val, y_test))
            std_scaler_X = std_scaler_y = scaler
    else:
        std_scaler_X = std_scaler_y = None
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), std_scaler_X, std_scaler_y

def create_ts_dl(X, y, lags, horizon, stride, batch_size, device, val_size, test_size, data_preprocess=("log_returns", "normalize"), drop_last=False, rate=1, scaler=None):
    """
    Full pipeline of building train / val / test torch dataloaders
        from original numpy arrays
    """
    if "log_returns" in data_preprocess:
        X = log_returns(X)
        y = log_returns(y)
    X, y = build_ts_X_y(X, y, lags=lags, horizon=horizon, stride=stride)

    (X_train, X_val, X_test), (y_train, y_val, y_test) = split_data(X, y.reshape((len(X), - 1)), val_size=val_size, test_size=test_size, rate=rate)
    if "normalize" in data_preprocess:
        if scaler is None:
            X_train, X_val, X_test, std_scaler_X = normalize(X_train, X_val, X_test)
            y_train, y_val, y_test, std_scaler_y = normalize(y_train, y_val, y_test)
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = map(scaler.transform, (X_train, X_val, X_test, y_train, y_val, y_test))
            std_scaler_X = std_scaler_y = scaler
    else:
        std_scaler_X = std_scaler_y = None
    X_train, X_val, X_test, y_train, y_val, y_test = map(lambda x: torch.from_numpy(x).float().to(device), (X_train, X_val, X_test, y_train, y_val, y_test))
    
    train_dl = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=False, drop_last=drop_last)
    val_dl = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False, drop_last=drop_last)
    test_dl = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False, drop_last=drop_last)
    
    return train_dl, val_dl, test_dl, std_scaler_X, std_scaler_y


class DimUniversalStandardScaler:
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.mu = np.mean(data)
        self.std = np.std(data)
    
    def transform(self, data):
        return (data - self.mu) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.std + self.mu
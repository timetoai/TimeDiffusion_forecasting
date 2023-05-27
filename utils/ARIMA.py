import pmdarima as pm
import numpy as np

def train_forecast(seq: np.array, horizon: int, pbar=False):
    """
    Trains TimeDiffusion model on sequence and forecast for horizon length
    """
    tmean = seq.mean()
    tstd = seq.std()
    train = (seq - tmean) / tstd

    arima = pm.auto_arima(train, error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=5,
                      seasonal=True, m=12)
    forecast = arima.predict(n_periods=horizon)

    return {"model": arima, "forecast": forecast}
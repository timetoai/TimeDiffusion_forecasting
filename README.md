# TimeDiffusion model for forecasting

Diffusion based temporal convolutional model for time-series forecasting. Main focus on long term time-series forecasting.

**Project structure**
 * [utils](./utils)
   * [utils/dl.py](./utils/dl.py) - time-series deep learning models in pytorch with some decorators for training / inference
   * [utils/data.py](./utils/data.py) - functions for datasets processing
   * [utils/timediffusion.py](./utils/timediffusion.py) - TimeDiffusion model class
 * [results](./results) - folder with csv results files of model evaluation on different datasets
 * [eval_td.py](./eval_td.py) - python script for full evaluation pipeline on chosen dataset
 * [TimeDIffussion_exchange_rate_tests.ipynb](./TimeDIffussion_exchange_rate_tests.ipynb) - visualizations of the forecasts
 * [TD_vis.ipynb](./TD_vis.ipynb) - different visualizations
 

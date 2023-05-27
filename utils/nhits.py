import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, RMSE, MQF2DistributionLoss

def nhits_train_forecast(data, training_part, validation_part, context_length, prediction_length,
                          batch_size=128, max_epochs=5, return_metrics=False, SEED=42):
    warnings.filterwarnings("ignore")
    pl.seed_everything(SEED)
    subseq_len = context_length + prediction_length
    if validation_part - training_part < subseq_len:
        training_part = validation_part - subseq_len - 1
    if len(data) - validation_part < subseq_len:
        validation_part = len(data) - subseq_len - 1
    
    test_data = data[data.time_idx > validation_part]
    train_data = data[data.time_idx <= validation_part]

    training = TimeSeriesDataSet(
        train_data[train_data.time_idx <= training_part],
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    testing = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        predict_mode=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, train_data, min_prediction_idx=training_part + 1)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        enable_model_summary=False,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback],
        limit_train_batches=30,
        enable_checkpointing=True,
    )
    net = NHiTS.from_dataset(
        training,
        learning_rate=5e-3,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        hidden_size=64,
        optimizer="AdamW",
        loss=MQF2DistributionLoss(prediction_length=prediction_length),
    )
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path)
    res = best_model.predict(test_dataloader, return_y=True)
    if return_metrics:
        return {"mae": MAE()(res.output, res.y), "mse": RMSE()(res.output, res.y) ** 2}
    return res
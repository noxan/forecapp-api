import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import json
from neuralprophet import NeuralProphet
import numpy as np
import pandas as pd
import plotly
from app.config import Dataset, ModelConfig
import multiprocessing as mp


def run_prediction(
    dataset: Dataset,
    configuration: ModelConfig,
    training_callback: Callback | None = None,
):
    items = [item.dict() for item in dataset.__root__]
    print("dataset", "n=" + str(len(items)))
    df = pd.DataFrame(items)
    df = df.dropna()
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"])

    print(df.head())
    print(df.dtypes)
    config = configuration

    is_autoregression = config.autoregression.lags > 0

    training_cfg = (
        {"callbacks": [training_callback]} if training_callback != None else {}
    )

    m = NeuralProphet(
        n_forecasts=config.forecasts if is_autoregression else 1,
        n_lags=config.autoregression.lags,
        # trend
        growth=config.trend.growth,
        n_changepoints=config.trend.number_of_changepoints,
        # seasonality
        yearly_seasonality=config.seasonality.yearly,
        weekly_seasonality=config.seasonality.weekly,
        daily_seasonality=config.seasonality.daily,
        seasonality_mode=config.seasonality.mode,
        seasonality_reg=config.seasonality.regularization,
        # training
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        # training callback
        trainer_config=training_cfg,
    )

    for lagged_regressor in config.lagged_regressors:
        m.add_lagged_regressor(
            lagged_regressor.name,
            n_lags=lagged_regressor.lags,
            regularization=lagged_regressor.regularization,
            normalize=lagged_regressor.normalize,
        )

    metrics = m.fit(
        df,
        checkpointing=False,
        progress=None,
        freq=config.frequency,
        early_stopping=config.training.early_stopping,
    )
    metrics = metrics if metrics is not None else pd.DataFrame()

    df_future = m.make_future_dataframe(
        df,
        n_historic_predictions=True,
        periods=config.forecasts,
    )

    fcst = m.predict(df_future)
    print("fcst", fcst.columns)

    # TODO: Map all dataframes to the same format (best with proper names already)
    if is_autoregression:
        # Values n_lags+: ds, y, yhat1, yhat2, ..., ar1, ar2, ..., trend, season_weekly, season_daily
        # Values latest: ds, y, origin0
        fcst_latest = m.get_latest_forecast(fcst, include_history_data=True)
        df_fcst = pd.concat([fcst, fcst_latest], axis=1)
        print("df_fcst", df_fcst.columns)
        # Remove autoregression columns from response
        filtered_columns = [
            c
            for c in df_fcst.columns
            if not c.startswith("ar")
            and not (c.startswith("lagged_regressor") and not c.endswith("1"))
            and (not c.startswith("yhat") or c == "yhat1")
        ]
        # TODO: The autoregression df_fcst has currently yhat1 for the historic forecast and yhat2 for the future forecast, while the non-autoregression df_fcst has yhat1 which contains both historic and future forecast.
        print("df_fcst", filtered_columns)
        df_fcst = df_fcst[filtered_columns]
        # Merge origin-0 (the future prediction) into yhat1 (the historic prediction)
        # df_fcst = df_fcst.rename(columns={"origin-0": "yhat2"})
        df_fcst["yhat1"] = df_fcst["yhat1"].fillna(df_fcst["origin-0"])
        filtered_columns.remove("origin-0")
        df_fcst = df_fcst[filtered_columns]
    else:
        # Values default: ds, y, yhat1, trend, season_yearly, season_weekly, season_daily
        df_fcst = fcst
    # TODO: Sort df_fcst columns by name or something

    fig = m.plot_parameters(plotting_backend="plotly")
    # fig2 = m.plot_components(fcst, plotting_backend="plotly")

    return {
        "status": "ok",
        "configuration": json.loads(configuration.json()),
        "forecast": json.loads(df_fcst.replace({np.nan: None}).to_json()),
        "metrics": json.loads(metrics.replace({np.nan: None}).to_json()),
        "explainable": {
            "parameters": json.loads(plotly.io.to_json(fig)),
            # "components": json.loads(plotly.io.to_json(fig2)),
        },
    }


class NPProgressCallback(Callback):
    def __init__(self, training_started: mp.Event, progress, is_done):
        self.progress = progress
        self.is_done = is_done
        self.is_started = training_started

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        self.progress.value = (trainer.current_epoch * 100) // trainer.max_epochs

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.is_done.value = True

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.is_started.set()


class NPProcess(mp.Process):
    def __init__(
        self,
        dataset: Dataset,
        config: ModelConfig,
        training_started: mp.Event,
        progress,
        is_done,
        pipe,
    ):
        self.dataset = dataset
        self.config = config
        self.callback = NPProgressCallback(training_started, progress, is_done)
        self.pipe = pipe
        self.training_started = training_started
        mp.Process.__init__(self)

    def run(self):
        result = run_prediction(self.dataset, self.config, self.callback)
        self.pipe.send(result)

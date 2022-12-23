import numpy as np
import pandas as pd
from pydantic import Field
import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neuralprophet import NeuralProphet, np_types, set_log_level
from pydantic import BaseModel, Extra

sentry_sdk.init(
    dsn="https://5849277f70ea4dbdba8ce47bbbe1b552@o4504138709139456.ingest.sentry.io/4504138710253568",
    # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring.
    # We recommend adjusting this value in production,
    traces_sample_rate=1.0,
)

set_log_level("WARNING")


class DatasetItem(BaseModel, extra=Extra.allow):
    ds: str | int
    y: str | float | int


class Dataset(BaseModel):
    __root__: list[DatasetItem] = Field(..., min_items=1)


class TrainingConfig(BaseModel):
    epochs: int | None = None


class AutoregressionConfig(BaseModel):
    lags: int = 0
    regularization: float = 0.0


class ModelConfig(BaseModel):
    forecasts: int = 1
    frequency: str = "auto"
    autoregression: AutoregressionConfig = AutoregressionConfig()
    yearly_seasonality: np_types.SeasonalityArgument = False
    training: TrainingConfig


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def prediction(dataset: Dataset, configuration: ModelConfig):
    config = configuration
    print(config)

    items = [item.dict() for item in dataset.__root__]
    print("dataset", "n=" + str(len(items)))
    df = pd.DataFrame(items)
    df = df.dropna()
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"])

    print(df.head())
    print(df.dtypes)

    m = NeuralProphet(
        n_forecasts=config.forecasts,
        n_lags=config.autoregression.lags,
        yearly_seasonality=config.yearly_seasonality,
        epochs=3,
    )

    metrics = m.fit(df, checkpointing=False, progress=None, freq=config.frequency)
    metrics = metrics if metrics is not None else pd.DataFrame()

    fcst = m.predict(df)
    print("fcst", fcst.columns)
    # Values default: ds, y, yhat1, trend, season_yearly, season_weekly, season_daily
    # Values n_lags+: ds, y, yhat1, yhat2, ar1, ar2, trend, season_weekly, season_daily

    if config.autoregression.lags > 0:
        # Values latest: ds, y, origin0
        df_fcst = m.get_latest_forecast(fcst)
        print("df_fcst", df_fcst.columns)
    else:
        df_fcst = fcst

    return {
        "status": "ok",
        "forecast": fcst.replace({np.nan: None}).to_dict(),
        "metrics": metrics.replace({np.nan: None}).to_dict(),
    }

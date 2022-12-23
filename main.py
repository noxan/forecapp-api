import pandas as pd
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


class TrainingConfig(BaseModel):
    epochs: int | None = None


class ModelConfig(BaseModel):
    forecasts: int = 1
    autoregression_lags: int = 0
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
def prediction(dataset: list[DatasetItem], configuration: ModelConfig):
    config = configuration
    print(config)

    items = [item.dict() for item in dataset]
    print("dataset", "n=" + str(len(items)))
    print(items[0])
    df = pd.DataFrame(items)
    df = df.dropna()
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"])

    print(df.head())
    print(df.dtypes)

    m = NeuralProphet(
        n_forecasts=config.forecasts,
        n_lags=config.autoregression_lags,
        yearly_seasonality=config.yearly_seasonality,
        epochs=1,
    )

    metrics = m.fit(df, checkpointing=False, progress=None)
    metrics = metrics if metrics is not None else pd.DataFrame()

    fcst = m.predict(df)

    df_fcst = m.get_latest_forecast(fcst)

    return {
        "status": "ok",
        "config": config,
        "forecast": df_fcst.to_dict(),
        "metrics": metrics.to_dict(),
    }

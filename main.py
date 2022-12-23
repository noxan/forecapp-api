import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neuralprophet import NeuralProphet, np_types, set_log_level
from pydantic import BaseModel, Extra

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

    metrics = m.fit(df, checkpointing=False, progress=None) or pd.DataFrame()

    return {
        "status": "ok",
        "config": config,
        "metrics": metrics.to_dict(),
    }

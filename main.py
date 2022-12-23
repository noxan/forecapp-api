import pandas as pd
from fastapi import FastAPI
from neuralprophet import NeuralProphet, np_types, set_log_level
from pydantic import BaseModel

set_log_level("WARNING")


class DatasetItem(BaseModel):
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def prediction(dataset: list[DatasetItem], configuration: ModelConfig):
    config = configuration
    print("dataset", len(dataset))
    print(config)

    df = pd.DataFrame(dataset)

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

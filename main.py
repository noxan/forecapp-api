import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from neuralprophet import NeuralProphet, np_types


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
def prediction(dataset: list, config: ModelConfig):
    print(dataset)
    print(config)

    df = pd.DataFrame(dataset)

    m = NeuralProphet(
        n_forecasts=config.forecasts,
        n_lags=config.autoregression_lags,
        yearly_seasonality=config.yearly_seasonality,
    )

    return {"status": "ok", "config": config}

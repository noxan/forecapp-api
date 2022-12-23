from fastapi import FastAPI
from pydantic import BaseModel

from neuralprophet import NeuralProphet, np_types


class ModelConfiguration(BaseModel):
    forecasts: int = 1
    autoregression_lags: int = 0
    yearly_seasonality: np_types.SeasonalityArgument = False
    training: bool = False


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def prediction(configuration: ModelConfiguration):
    print(configuration)

    m = NeuralProphet(
        n_forecasts=configuration.forecasts,
        n_lags=configuration.autoregression_lags,
        yearly_seasonality=configuration.yearly_seasonality,
    )

    return {"status": "ok", "configuration": configuration}

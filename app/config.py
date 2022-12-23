from neuralprophet import np_types
from pydantic import BaseModel, Extra, Field


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
    training: TrainingConfig = TrainingConfig()

from neuralprophet import np_types
from pydantic import BaseModel, Extra, Field
from fastapi_camelcase import CamelModel


class DatasetItem(BaseModel, extra=Extra.allow):
    ds: str | int
    y: str | float | int


class Dataset(BaseModel):
    __root__: list[DatasetItem] = Field(..., min_items=1)


class TrendConfig(CamelModel):
    growth: np_types.GrowthMode = "linear"
    number_of_changepoints: int = 10


class TrainingConfig(CamelModel):
    epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    early_stopping: bool = True


class AutoregressionConfig(BaseModel):
    lags: int = 0
    regularization: float = 0.0


class SeasonalityConfig(BaseModel):
    yearly: np_types.SeasonalityArgument = "auto"
    weekly: np_types.SeasonalityArgument = "auto"
    daily: np_types.SeasonalityArgument = "auto"
    mode: np_types.SeasonalityMode = "additive"
    regularization: float = 0


class LaggedRegressorConfig(BaseModel):
    name: str


class ModelConfig(CamelModel):
    forecasts: int = 1
    frequency: str = "auto"
    trend: TrendConfig = TrendConfig()
    autoregression: AutoregressionConfig = AutoregressionConfig()
    seasonality: SeasonalityConfig = SeasonalityConfig()
    training: TrainingConfig = TrainingConfig()
    lagged_regressors: list[LaggedRegressorConfig] = []

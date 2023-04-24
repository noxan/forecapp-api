from typing import List, Union

from fastapi_camelcase import CamelModel
from neuralprophet import np_types
from pydantic import BaseModel, Extra, Field


class DatasetItem(BaseModel, extra=Extra.allow):
    ds: Union[str, int]
    y: Union[str, float, int]


class Dataset(BaseModel):
    __root__: List[DatasetItem] = Field(..., min_items=1)


class TrendConfig(CamelModel):
    growth: np_types.GrowthMode = "linear"
    number_of_changepoints: int = 10


class TrainingConfig(CamelModel):
    epochs: Union[int, None] = None
    learning_rate: Union[float, None] = None
    batch_size: Union[int, None] = None
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
    lags: int
    regularization: float
    normalize: bool | str



class ModelConfig(CamelModel):
    forecasts: int = Field(default=1, ge=1)
    frequency: str = "auto"
    trend: TrendConfig = TrendConfig()
    autoregression: AutoregressionConfig = AutoregressionConfig()
    seasonality: SeasonalityConfig = SeasonalityConfig()
    training: TrainingConfig = TrainingConfig()
    lagged_regressors: List[LaggedRegressorConfig] = []

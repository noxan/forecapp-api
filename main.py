import json

import numpy as np
import pandas as pd
import plotly
import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neuralprophet import NeuralProphet, set_log_level

from app.config import Dataset, ModelConfig
from app.events import create_event_dataframe

sentry_sdk.init(
    dsn="https://5849277f70ea4dbdba8ce47bbbe1b552@o4504138709139456.ingest.sentry.io/4504138710253568",
    # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring.
    # We recommend adjusting this value in production,
    traces_sample_rate=1.0,
)

set_log_level("WARNING")


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
    print("config", config)

    items = [item.dict() for item in dataset.__root__]
    print("dataset", "n=" + str(len(items)))
    df = pd.DataFrame(items)
    df = df.dropna()
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"])

    print(df.head())
    print(df.dtypes)

    is_autoregression = config.autoregression.lags > 0

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
    )

    first = True
    # add events
    if config.events:
        for event_name, event in config.events.items():
            event_df = create_event_dataframe(event_name, event)
            if first:
                events_df = event_df
                first = False
            else:
                events_df = pd.concat([events_df, event_df], axis=1)
            m.add_events(
                list(event_name),
                lower_window=event.lowerWindow,
                upper_window=event.upperWindow,
                regularization=event.regularization,
                mode=event.mode,
            )
        df = m.create_df_with_events(df, events_df)

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
        "configuration": configuration,
        "forecast": df_fcst.replace({np.nan: None}).to_dict(),
        "metrics": metrics.replace({np.nan: None}).to_dict(),
        "explainable": {
            "parameters": json.loads(plotly.io.to_json(fig)),
            # "components": json.loads(plotly.io.to_json(fig2)),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

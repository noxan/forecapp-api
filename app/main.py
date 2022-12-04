import pandas
import numpy as np
import sentry_sdk
from flask import Flask, jsonify, request
from flask_cors import CORS
from neuralprophet import NeuralProphet, set_log_level
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="https://5849277f70ea4dbdba8ce47bbbe1b552@o4504138709139456.ingest.sentry.io/4504138710253568",
    integrations=[
        FlaskIntegration(),
    ],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
)


set_log_level("ERROR")

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/prediction", methods=["POST"])
def prediction():
    payload = request.get_json(force=True)
    if "dataset" not in payload:
        return jsonify({"error": "dataset is missing"}), 400
    if "configuration" not in payload:
        return jsonify({"error": "model configuration is missing"}), 400

    dataset = payload["dataset"]
    configuration = payload["configuration"]

    print("model configuration", configuration)

    epochs = configuration.get("training", {}).get("epochs", None)
    # Map "auto" for epochs back to None... else it all fails
    if epochs == "auto":
        epochs = None

    df = pandas.DataFrame(dataset)
    print(df.dtypes)
    print(df.head())

    # TODO: those transforms should not be necessary in the future
    df = df.dropna()

    # Test for unix timestamp
    try:
        int(df["ds"][0])
    except ValueError:
        df["ds"] = pandas.to_datetime(df["ds"])
    else:
        df["ds"] = pandas.to_datetime(df["ds"], unit="s")
    df["ds"] = df["ds"].dt.tz_localize(None)
    df["y"] = pandas.to_numeric(df["y"])

    model = NeuralProphet(epochs=epochs)

    if "countryHolidays" in configuration:
        for country in configuration["countryHolidays"]:
            print("Add country holidays for", country)
            model = model.add_country_holidays(country_name=country)

    for lagged_regressor in configuration.get("laggedRegressors", []):
        name = lagged_regressor["dataColumnRef"]
        print("Add lagged regressor", name)
        df[name] = pandas.to_numeric(df[name])
        n_lags = lagged_regressor.get("n_lags", "auto")
        if n_lags != "auto" and n_lags != "scalar":
            n_lags = int(n_lags)
        model = model.add_lagged_regressor(
            name,
            n_lags,
            lagged_regressor.get("regularization", None),
            lagged_regressor.get("normalize", "auto"),
        )

    metrics = model.fit(df, freq="D")

    forecast = model.predict(df)

    # Convert time to unix timestamp
    # forecast["ds"] = forecast["ds"].astype(int) / 10**9

    return jsonify(
        {
            "status": "ok",
            "metrics": metrics.replace({np.nan: None}).to_dict("tight"),
            "forecast": forecast.replace({np.nan: None}).to_dict("records"),
        }
    )

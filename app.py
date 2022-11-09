import pandas
from flask import Flask, request, jsonify
from neuralprophet import NeuralProphet, set_log_level
from flask_cors import CORS

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

    epochs = configuration.get("training", {}).get("epochs", None)

    df = pandas.DataFrame(dataset)
    print(df.dtypes)
    print(df.head())

    # TODO: those transforms should not be necessary in the future
    df = df.dropna()
    df["ds"] = pandas.to_datetime(df["ds"])
    df["y"] = pandas.to_numeric(df["y"])

    model = NeuralProphet(epochs=epochs)
    metrics = model.fit(df, freq="D")

    forecast = model.predict(df)

    return jsonify(
        {
            "status": "ok",
            "metrics": metrics.to_dict(),
            "forecast": forecast.to_dict("records"),
        }
    )

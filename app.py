import json
from flask import Flask, request, jsonify
from neuralprophet import NeuralProphet

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/prediction", methods=["POST"])
def prediction():
    content = request.get_json(force=True)
    # model = NeuralProphet()
    # print(model)
    return jsonify({"message": "Hello, World!"})

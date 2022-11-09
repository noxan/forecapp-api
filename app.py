from flask import Flask, request, jsonify
from neuralprophet import NeuralProphet
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/prediction", methods=["POST"])
def prediction():
    content = request.get_json(force=True)
    # model = NeuralProphet()
    # print(model)
    return jsonify({"message": "Hello, World!"})

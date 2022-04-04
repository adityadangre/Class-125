from flask import Flask, jsonify, request
from classifier import prediction
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello world"

@app.route("/predict-digit", methods=["POST"])
def predict_digit():
    image = request.files.get("digit")
    predict = prediction(image)
    return jsonify({
        "prediction": predict
    })

if __name__ == "__main__":
    app.run(debug=True)
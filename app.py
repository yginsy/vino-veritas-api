
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# loading model
model = joblib.load("wine_model.pkl")
scaler = joblib.load("wine_scaler.pkl")

cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

@app.route("/")
def home():
    return "api running"

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.get_json()

    df = pd.DataFrame([data], columns=cols)

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]

    if pred == 1:
        result = "Good Quality"
    else:
        result = "Bad Quality"

    return jsonify({"quality_prediction": result})

if __name__ == "__main__":
    app.run(debug=True)

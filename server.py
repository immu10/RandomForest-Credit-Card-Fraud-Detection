import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Sample generated transaction


def calc(data):

    new_transaction_data = pd.DataFrame({key: [value] for key, value in data.items()})

    features = ['transaction_id', 'hour_of_day', 'category', 'amount(usd)', 'merchant', 'job']
    new_transaction_data = new_transaction_data[features].set_index("transaction_id")
    enc = OrdinalEncoder(dtype=np.int64)
    enc.fit(new_transaction_data.loc[:, ['category','merchant','job']])

    new_transaction_data.loc[:, ['category','merchant','job']] = enc.transform(new_transaction_data[['category','merchant','job']])


    try:
        fraud_proba = model.predict_proba(new_transaction_data)[:, 1]  # Probability of fraud
        fraud_pred = model.predict(new_transaction_data)              # Fraud prediction (0 or 1)
        print("Prediction Results:")
        print(fraud_proba)
        print(fraud_pred)
        # Display results
        # new_transaction_data["Fraud_Proba"] = fraud_proba
        # new_transaction_data["Fraud_Predict"] = fraud_pred
        return {
            'fraud_proba': fraud_proba.tolist(),
            "fraud_pred" : fraud_pred.tolist()
        }

    #     
    except Exception as e:
        print(f"Error during prediction: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()  # Ensure the data is in JSON format
    print(data)
    # Example processing (e.g., running a model)
    # Assuming the model is loaded and called `model`
    # response = model.predict(data)

    # For simplicity, we'll return the data with a success message
    return jsonify({
        "message": "POST request received successfully!",
        "received_data": data,
        "prediction": calc(data)
    })
@app.route("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Runs on http://127.0.0.1:5000
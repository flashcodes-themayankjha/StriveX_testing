from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load trained models
fitness_model = joblib.load("fitness_model.joblib")
risk_model = joblib.load("risk_model.joblib")

# Define Flask app
app = Flask(__name__)

# API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input values
        strength = data['strength']
        stamina = data['stamina']
        endurance = data['endurance']
        eye_sight = data['eye_sight']
        gender = data['gender']
        sport = data['sport']

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "Strength": strength,
            "Stamina": stamina,
            "Endurance": endurance,
            "Eye_Sight": eye_sight,
            "Gender": gender,
            "Sport": sport
        }])

        # Predict
        fitness_pred = fitness_model.predict(input_data)[0]
        risk_pred = risk_model.predict(input_data)[0]

        return jsonify({
            "fitness_score": round(float(fitness_pred), 2),
            "injury_risk": round(float(risk_pred), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

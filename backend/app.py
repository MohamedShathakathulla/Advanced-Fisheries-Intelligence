from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        loc = data.get("location")
        species = data.get("species")
        time_range = data.get("timeRange")

        # Encode input
        input_data = {
            "location": encoders["location"].transform([loc])[0],
            "species": encoders["species"].transform([species])[0],
            "time_range": encoders["time_range"].transform([time_range])[0]
        }

        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        success_rate = min(100, max(30, int((prediction / 100) * 100)))

        return jsonify({
            "catch_volume": round(prediction, 2),
            "success_rate": success_rate,
            "recommendation": "Try early morning or late evening for higher success."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

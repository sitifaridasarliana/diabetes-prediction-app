from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model_diabetes.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = np.array([[
        float(request.form["Glucose"]),
        float(request.form["BMI"]),
        float(request.form["Age"]),
        float(request.form["BloodPressure"]),
        float(request.form["Insulin"]),
        float(request.form["DiabetesPedigreeFunction"]),
        float(request.form["physical_activity_minutes_per_week"]),
        float(request.form["diet_score"]),
        float(request.form["sleep_hours_per_day"]),
        float(request.form["family_history_diabetes"])
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    result = (
        "Berisiko Diabetes"
        if prediction[0] == 1
        else "Tidak Berisiko Diabetes"
    )

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run()

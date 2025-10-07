from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iot.pkl")
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predic", methods=["GET","POST"])
def predic_page():
    prediction = None
    probability = None

    if request.method == "POST":
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        water_level = float(request.form.get("water_level"))
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))

        input_data = np.array([[temperature, humidity, water_level, N, P, K]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()

        prediction = "เปิด" if prediction == 1 else "ปิด"
        probability = f"{probability * 100:.2f}%"
    return render_template("prediction.html")
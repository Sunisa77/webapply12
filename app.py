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

@app.route("/predic")
def predic_page():
    return render_template("prediction.html")
import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv("Bengaluru_House_Data_Cleaned.csv")
pipe = pickle.load(open("RidgeHouseModel.pkl", "rb"))

@app.route("/")
def index():
    locations = sorted(data["location"].unique())
    return render_template("index.html", locations = locations)


@app.route("/predict", methods = ["POST"])
def predict():
    location = request.form.get("location")
    bhk = float(request.form.get("bhk"))
    bath = float(request.form.get("bath"))
    sqft = request.form.get("sqft")

    inputs = pd.DataFrame([[location, sqft, bath, bhk]], columns = ["location", "total_sqft", "bath", "bhk"])
    prediction = pipe.predict(inputs)[0] * 100000

    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug = True)
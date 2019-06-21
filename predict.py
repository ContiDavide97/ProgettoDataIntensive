import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    inputs = [
        float(request.args["RAINFALL"]),
        float(request.args["WIND_GUST_SPEED"]),
        float(request.args["HUMIDITY_3PM"]),
        float(request.args["PRESSURE_3PM"]),
        int(request.args["RAIN_TODAY"])
    ]
    with app.open_resource("model.bin", "rb") as f:
        model = pickle.load(f)
    output = model.predict_proba([inputs])
    rainProb = output[:, 1][0]
    noRainProb = output[0, :][0]
    response = ("Domani pioverà con probabilità: " + str(int(rainProb * 100)) + "%") if rainProb >= 0.50 else ("Domani non pioverà con probabilità: " + str(int(noRainProb * 100)) + "%")
    return render_template("predict.html", resp=response)
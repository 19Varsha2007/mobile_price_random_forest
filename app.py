from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("mobile_price.csv")

# Features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# ==============================
# Create and Train Random Forest
# ==============================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# ==============================
# Home Page
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        battery_power = float(request.form["battery_power"])
        blue = float(request.form["blue"])
        clock_speed = float(request.form["clock_speed"])
        dual_sim = float(request.form["dual_sim"])
        fc = float(request.form["fc"])
        four_g = float(request.form["four_g"])
        int_memory = float(request.form["int_memory"])
        m_dep = float(request.form["m_dep"])
        mobile_wt = float(request.form["mobile_wt"])
        n_cores = float(request.form["n_cores"])
        pc = float(request.form["pc"])
        px_height = float(request.form["px_height"])
        px_width = float(request.form["px_width"])
        ram = float(request.form["ram"])
        sc_h = float(request.form["sc_h"])
        sc_w = float(request.form["sc_w"])
        talk_time = float(request.form["talk_time"])
        three_g = float(request.form["three_g"])
        touch_screen = float(request.form["touch_screen"])
        wifi = float(request.form["wifi"])

        # Create input DataFrame
        input_data = pd.DataFrame([[
            battery_power,
            blue,
            clock_speed,
            dual_sim,
            fc,
            four_g,
            int_memory,
            m_dep,
            mobile_wt,
            n_cores,
            pc,
            px_height,
            px_width,
            ram,
            sc_h,
            sc_w,
            talk_time,
            three_g,
            touch_screen,
            wifi
        ]], columns=X.columns)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Price range names
        price_ranges = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
        }

        result = price_ranges.get(
            int(prediction),
            str(prediction)
        )

        return render_template(
            "index.html",
            prediction=result
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error: " + str(e)
        )


# ==============================
# Run Flask App
# ==============================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )

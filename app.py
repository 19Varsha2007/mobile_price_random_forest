from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("mobile_price.csv")

# Separate input features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []

        # Get all input values
        for column in X.columns:
            value = float(request.form[column])
            values.append(value)

        # Create input dataframe
        input_data = pd.DataFrame(
            [values],
            columns=X.columns
        )

        # Predict
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


# Run application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )

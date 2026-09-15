from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("mobile_price.csv")

# Separate features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        values = []

        for column in X.columns:
            values.append(float(request.form[column]))

        # Create input dataframe
        new_mobile = pd.DataFrame(
            [values],
            columns=X.columns
        )

        # Prediction
        result = model.predict(new_mobile)[0]

        price_ranges = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
        }

        prediction = price_ranges[result]

    return render_template(
        "index.html",
        columns=X.columns,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
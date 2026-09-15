import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("mobile_price.csv")

# Display dataset
print("Mobile Price Dataset")
print("--------------------")
print(data.head())

# Separate input features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------
# Mobile Price Prediction
# --------------------------------

print("\nEnter Mobile Phone Details")

battery_power = float(input("Battery Power: "))
blue = int(input("Bluetooth (0/1): "))
clock_speed = float(input("Clock Speed: "))
dual_sim = int(input("Dual SIM (0/1): "))
fc = int(input("Front Camera: "))
four_g = int(input("4G (0/1): "))
int_memory = int(input("Internal Memory: "))
m_dep = float(input("Mobile Depth: "))
mobile_wt = int(input("Mobile Weight: "))
n_cores = int(input("Number of Cores: "))
pc = int(input("Primary Camera: "))
px_height = int(input("Pixel Height: "))
px_width = int(input("Pixel Width: "))
ram = int(input("RAM: "))
sc_h = int(input("Screen Height: "))
sc_w = int(input("Screen Width: "))
talk_time = int(input("Talk Time: "))
three_g = int(input("3G (0/1): "))
touch_screen = int(input("Touch Screen (0/1): "))
wifi = int(input("WiFi (0/1): "))

# Create input data
new_mobile = pd.DataFrame([[
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

# Predict price range
prediction = model.predict(new_mobile)[0]

# Price range labels
price_ranges = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("\n================================")
print("Predicted Mobile Price Range:")
print(price_ranges[prediction])
print("================================")
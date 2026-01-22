from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# -------------------------
# Load & Train Model
# -------------------------
data = pd.read_csv("diabetes.csv")

# Add Sex column (0 = Female, 1 = Male)
data["Sex"] = 0

cols = ["Glucose", "BloodPressure", "Insulin", "BMI"]
for col in cols:
    data[col] = data[col].replace(0, data[col].mean())

X = data[
    ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age", "Sex"]
]
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    sex = int(data["sex"])
    pregnancies = int(data["pregnancies"]) if sex == 0 else 0

    input_df = pd.DataFrame([[
        pregnancies,
        float(data["glucose"]),
        float(data["bp"]),
        float(data["insulin"]),
        float(data["bmi"]),
        int(data["age"]),
        sex
    ]], columns=X.columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    risk = int(probability * 100)

    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return jsonify({
        "result": result,
        "risk": risk
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)


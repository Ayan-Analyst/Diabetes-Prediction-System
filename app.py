import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit.components.v1 as components


# Page Config

st.set_page_config(page_title="Diabetes Prediction System", layout="wide")


# Load & Train Model

data = pd.read_csv("diabetes.csv")

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    data[col] = data[col].replace(0, data[col].mean())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# HTML + CSS + JavaScript UI

html_code = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Diabetes Prediction</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: linear-gradient(to right, #e3f2fd, #ffffff);
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1100px;
    margin: auto;
    padding: 30px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    color: #1f4e79;
    font-size: 40px;
}

.header p {
    font-size: 18px;
    color: #444;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
}

label {
    font-weight: bold;
    margin-top: 10px;
}

input {
    width: 100%;
    padding: 8px;
    margin-top: 5px;
    border-radius: 6px;
    border: 1px solid #ccc;
}

button {
    margin-top: 20px;
    padding: 12px;
    width: 100%;
    background-color: #1f77b4;
    color: white;
    border: none;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
}

button:hover {
    background-color: #155a8a;
}

.footer {
    margin-top: 30px;
    text-align: center;
    color: #555;
    font-size: 14px;
}
</style>

<script>
function showMessage() {
    alert("Scroll down to see prediction result below 👇");
}
</script>

</head>

<body>
<div class="container">

<div class="header">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>Early diabetes detection using Machine Learning</p>
</div>

<div class="card">
    <div class="grid">
        <div>
            <label>Pregnancies</label>
            <input id="p1" type="number" value="1">

            <label>Glucose Level</label>
            <input id="p2" type="number" value="120">

            <label>Blood Pressure</label>
            <input id="p3" type="number" value="70">

            <label>Skin Thickness</label>
            <input id="p4" type="number" value="25">
        </div>

        <div>
            <label>Insulin</label>
            <input id="p5" type="number" value="100">

            <label>BMI</label>
            <input id="p6" type="number" value="30">

            <label>Diabetes Pedigree Function</label>
            <input id="p7" type="number" step="0.01" value="0.5">

            <label>Age</label>
            <input id="p8" type="number" value="35">
        </div>
    </div>

    <button onclick="showMessage()">Predict Diabetes</button>
</div>

<div class="footer">
    Project Type: Supervised Learning (Classification) |
    Algorithm: Decision Tree |
    Use Case: Early Diabetes Detection
</div>

</div>
</body>
</html>
"""

components.html(html_code, height=720)


# Prediction Section 

st.markdown("## 🔍 Prediction Result")

if st.button("Run Prediction"):
    sample = [1, 120, 70, 25, 100, 30, 0.5, 35]
    input_df = pd.DataFrame([sample], columns=X.columns)
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("🟥 Patient is DIABETIC")
    else:
        st.success("🟩 Patient is NON-DIABETIC")

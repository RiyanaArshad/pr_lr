import numpy as np
import pandas as pd
import joblib  # For saving models
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("pr_lr_dataset.csv")  # Replace with actual dataset

# Train Polynomial Regression Model
def train_polynomial_regression(feature_column, target_column, degree=2):
    X = df[[feature_column]].values
    y = df[target_column].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)

    joblib.dump((poly, model), "polynomial_model.pkl")
    return model

# Train Logistic Regression Model
def train_logistic_regression(feature_columns, target_column):
    X = df[feature_columns].values
    y = df[target_column].values

    if len(set(y)) < 2:  # Ensure there are at least two classes
        raise ValueError("Logistic Regression requires at least two classes in the target column.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump((scaler, model), "logistic_model.pkl")
    return model

# Train & Save Models
train_polynomial_regression("Advertising Budget ($)", "Sales Revenue ($)", degree=2)
train_logistic_regression(["Customer Age", "Credit Score"], "Purchased (0/1)")

# Serve the HTML Page
@app.route("/")
def home():
    return render_template("index.html")

# Handle Polynomial Regression Form Submission
@app.route("/predict/polynomial", methods=["POST"])
def predict_polynomial():
    feature_value = float(request.form["feature"])

    poly, model = joblib.load("polynomial_model.pkl")
    X_poly = poly.transform([[feature_value]])
    prediction = model.predict(X_poly)[0]

    return f"<h2>Predicted Sales Revenue: ${prediction:.2f}</h2> <a href='/'>Go Back</a>"

# Handle Logistic Regression Form Submission
@app.route("/predict/logistic", methods=["POST"])
def predict_logistic():
    age = float(request.form["age"])
    credit = float(request.form["credit"])

    scaler, model = joblib.load("logistic_model.pkl")
    X_scaled = scaler.transform([[age, credit]])
    proba = model.predict_proba(X_scaled)[0]

    threshold = 0.3  # Adjust if needed
    prediction = 1 if proba[1] > threshold else 0
    result = "Yes" if prediction == 1 else "No"

    return f"<h2>Will the customer purchase? {result}</h2> <a href='/'>Go Back</a>"


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

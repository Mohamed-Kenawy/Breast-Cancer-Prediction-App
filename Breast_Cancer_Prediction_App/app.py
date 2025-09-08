import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------
# Load data
# -----------------------------
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# -----------------------------
# Train or load model
# -----------------------------
try:
    # Load trained model + scaler if available
    with open("best_svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    # Train a new model if files not found
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(probability=True, kernel="rbf", random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    with open("best_svm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🧬 Breast Cancer Prediction App")

st.write("This app predicts whether a tumor is **Benign** or **Malignant** based on input features.")

# Sidebar for inputs
st.sidebar.header("Input Features")

def user_input_features():
    features = {}
    for col in X.columns:
        features[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction")
st.write("🔴 Malignant" if prediction[0] == 0 else "🟢 Benign")

st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=["Malignant", "Benign"]))

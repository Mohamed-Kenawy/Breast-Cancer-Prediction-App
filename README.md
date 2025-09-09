# 🧬 Breast Cancer Prediction App

This is a **Streamlit web application** that predicts whether a breast tumor is **Benign** or **Malignant** based on user-provided input features.  
The model is trained on the **Breast Cancer Wisconsin Dataset** from `scikit-learn`.

---

## 🚀 Features
- Interactive **sliders** in the sidebar for inputting tumor features.
- Displays **user input data** before prediction.
- Predicts if the tumor is **🟢 Benign** or **🔴 Malignant**.
- Shows **prediction probabilities** for both classes.
- Uses **Support Vector Machine (SVM)** with an **RBF kernel** for classification.
- Automatically **loads a saved model** (`best_svm_model.pkl`) and scaler (`scaler.pkl`) if available, otherwise trains a new one.

---

## 📂 Project Structure
```text
├── app.py               # Main Streamlit application
├── best_svm_model.pkl   # Trained SVM model (auto-saved after training)
├── scaler.pkl           # StandardScaler object (auto-saved after training)
└── README.md            # Project documentation
```


---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/breast-cancer-prediction.git
   cd breast-cancer-prediction

2.**Install dependencies**
   ```bash
pip install -r requirements.txt
 ```
or manually
```bash
pip install streamlit scikit-learn pandas numpy
```

3.**Run the app**
```bash
streamlit run app.py
```
or open on your browser from the link in description 

## 📊 Dataset
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Source:** Available in `scikit-learn` via `load_breast_cancer()`  
- **Features include:**  radius, texture, smoothness, compactness, symmetry, fractal dimension, etc.

---

## 🔮 Example Output
- **User Input Features:** Displayed in a table for review before prediction.  

- **Prediction Result:**  
  - 🟢 Benign  
  - 🔴 Malignant  

- **Prediction Probability:**  
  A DataFrame showing the probabilities for both classes (`Malignant` and `Benign`).  

# 🔴 Fraud Detection System (Machine Learning + Streamlit)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

An **AI-powered fraud detection web application** built with **Scikit-Learn and Streamlit**.  
The app allows users to train a machine learning model on transaction data and detect fraudulent activities with real-time risk assessment.

---

## 🚀 Features

- Upload CSV transaction datasets
- Automatic fraud label detection
- Train a **Random Forest Classifier**
- Fraud probability scoring
- Risk classification (Low / Medium / High)
- Confusion Matrix & ROC Curve
- Feature importance visualization
- Interactive, modern Streamlit UI

---

## 🧠 Machine Learning Pipeline

- **Model:** Random Forest Classifier
- **Preprocessing:**
  - Numeric feature selection
  - StandardScaler normalization
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

---

## 🖥️ Application Tabs

### 🚀 Train Model
Upload a CSV file containing transaction data and a fraud label (`fraud`, `label`, `is_fraud`, `isFraud`).

### 🔍 Detect Fraud
Upload new transaction data to receive:
- Fraud prediction
- Fraud probability
- Risk level assessment

### 📊 Performance
View:
- Classification report
- Confusion matrix
- ROC curve

### 📈 Analytics
Analyze feature importance driving fraud predictions.

---

## 📁 Dataset Requirements

- CSV format  
- Must include a fraud label column:
  - `fraud`
  - `label`
  - `is_fraud`
  - `isFraud`
- Numeric features only (non-numeric columns are ignored)

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-Learn  
- Pandas & NumPy  
- Matplotlib & Seaborn  

---

## 👤 Author

**Robert Marsh Deku**  
BA Political Science & Chinese – University of Ghana  

**Aspiring Data Scientist & AI Engineer**

**Interests:**
- Artificial Intelligence  
- Data Engineering  
- Applied Machine Learning  

---

## 📄 License

MIT License


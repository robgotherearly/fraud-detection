# 🔴 Fraud Detection System (Machine Learning + Streamlit)

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

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

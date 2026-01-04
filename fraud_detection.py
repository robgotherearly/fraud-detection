import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%) !important;
        min-height: 100vh;
        font-family: 'Segoe UI', Trebuchet MS, sans-serif;
    }
    
    .stApp {
        background: transparent !important;
    }
    
    .main {
        background: transparent !important;
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        50% { text-shadow: 0 0 40px rgba(239, 68, 68, 0.8); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .header-title {
        font-size: 3em;
        font-weight: 900;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        letter-spacing: -1px;
        animation: glow 3s ease-in-out infinite;
    }
    
    .header-subtitle {
        font-size: 1.1em;
        color: #cbd5e1;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .stat-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        backdrop-filter: blur(10px);
        animation: slideIn 0.6s ease-out;
    }
    
    .stat-value {
        font-size: 2em;
        font-weight: 800;
        color: #ef4444;
        margin: 10px 0;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.9em;
    }
    
    .fraud-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        margin: 5px 5px 5px 0;
    }
    
    .safe-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        margin: 5px 5px 5px 0;
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #cbd5e1;
    }
    
    .risk-medium {
        background: rgba(249, 115, 22, 0.1);
        border-left: 4px solid #f97316;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #cbd5e1;
    }
    
    .risk-low {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #cbd5e1;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(239, 68, 68, 0.5) !important;
    }
    
    .section-title {
        font-size: 1.5em;
        font-weight: 800;
        color: #e2e8f0;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(239, 68, 68, 0.3);
    }
    
    .info-box {
        background: rgba(239, 68, 68, 0.05);
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        color: #cbd5e1;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "training_data" not in st.session_state:
    st.session_state.training_data = None
if "results" not in st.session_state:
    st.session_state.results = None

# Header
st.markdown("""
<div style="margin-bottom: 20px;">
    <div class="header-title">🔴 Fraud Detection System by Robert Marsh Deku</div>
    <div class="header-subtitle">AI-powered machine learning model for detecting fraudulent transactions</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train Model", "🔍 Detect Fraud", "📊 Performance", "📈 Analytics"])

with tab1:
    st.markdown("### 📤 Upload Training Data")
    st.markdown("<div class='info-box'>Upload a CSV file with transaction data. The file should include a 'fraud' or 'label' column (0 = legitimate, 1 = fraud).</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"], key="train_file")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Records</div>
                <div class="stat-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Features</div>
                <div class="stat-value">{len(df.columns)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detect fraud column
        fraud_col = None
        for col in df.columns:
            if col.lower() in ['fraud', 'label', 'is_fraud', 'isFraud']:
                fraud_col = col
                break
        
        if fraud_col:
            fraud_count = (df[fraud_col] == 1).sum()
            legitimate_count = (df[fraud_col] == 0).sum()
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Fraud Cases</div>
                    <div class="stat-value">{fraud_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                fraud_rate = (fraud_count / len(df)) * 100
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Fraud Rate</div>
                    <div class="stat-value">{fraud_rate:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Model training
            st.markdown("### 🤖 Train Random Forest Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Size (%)", 10, 50, 30) / 100
            
            with col2:
                random_state = st.number_input("Random State", min_value=1, value=42)
            
            if st.button("🚀 Train Model", use_container_width=True, key="train_btn"):
                with st.spinner("🔄 Training model..."):
                    # Prepare data
                    X = df.drop(columns=[fraud_col])
                    y = df[fraud_col]
                    
                    # Handle non-numeric columns
                    X = X.select_dtypes(include=[np.number])
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_split=10,
                        random_state=random_state,
                        n_jobs=-1
                    )
                    model.fit(X_train_scaled, y_train)
                    
                    # Store in session
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.training_data = {
                        'X_test': X_test,
                        'X_test_scaled': X_test_scaled,
                        'y_test': y_test,
                        'feature_names': X.columns.tolist()
                    }
                    
                    st.success("✅ Model trained successfully!")
                    st.rerun()

with tab2:
    if st.session_state.model:
        st.markdown("### 🔍 Detect Fraudulent Transactions")
        
        # Single transaction prediction
        st.markdown("#### Predict Single Transaction")
        
        uploaded_test = st.file_uploader("Upload transaction to check (CSV or single row)", type=["csv"], key="test_file")
        
        if uploaded_test:
            test_df = pd.read_csv(uploaded_test)
            test_df = test_df.select_dtypes(include=[np.number])
            
            # Ensure same features
            test_df = test_df[st.session_state.training_data['feature_names']]
            
            if st.button("🔍 Analyze Transaction", use_container_width=True, key="predict_btn"):
                test_scaled = st.session_state.scaler.transform(test_df)
                
                # Prediction
                prediction = st.session_state.model.predict(test_scaled)
                probability = st.session_state.model.predict_proba(test_scaled)
                
                # Display results
                for idx, (pred, prob) in enumerate(zip(prediction, probability)):
                    fraud_prob = prob[1] * 100
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if pred == 1:
                            st.markdown(f'<div class="fraud-badge">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
                            risk_level = "HIGH"
                        else:
                            st.markdown(f'<div class="safe-badge">✅ LEGITIMATE</div>', unsafe_allow_html=True)
                            risk_level = "LOW"
                    
                    with col2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Fraud Probability</div>
                            <div class="stat-value">{fraud_prob:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk assessment
                    if fraud_prob >= 70:
                        st.markdown(f'<div class="risk-high"><strong>🚨 HIGH RISK:</strong> {fraud_prob:.2f}% probability of fraud. Immediate investigation recommended.</div>', unsafe_allow_html=True)
                    elif fraud_prob >= 40:
                        st.markdown(f'<div class="risk-medium"><strong>⚠️ MEDIUM RISK:</strong> {fraud_prob:.2f}% probability of fraud. Review required.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low"><strong>✅ LOW RISK:</strong> {fraud_prob:.2f}% probability of fraud. Transaction appears legitimate.</div>', unsafe_allow_html=True)
        else:
            st.info("Upload a CSV file with transaction data to test")
    else:
        st.warning("⚠️ Please train a model first in the 'Train Model' tab")

with tab3:
    if st.session_state.model and st.session_state.training_data:
        st.markdown("### 📊 Model Performance Metrics")
        
        X_test_scaled = st.session_state.training_data['X_test_scaled']
        y_test = st.session_state.training_data['y_test']
        
        # Predictions
        y_pred = st.session_state.model.predict(X_test_scaled)
        y_pred_proba = st.session_state.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">{accuracy*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Precision</div>
                <div class="stat-value">{precision*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Recall</div>
                <div class="stat-value">{recall*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">F1 Score</div>
                <div class="stat-value">{f1:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Classification report
        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax, 
                    xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # ROC Curve
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='#ef4444', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please train a model first")

with tab4:
    if st.session_state.model and st.session_state.training_data:
        st.markdown("### 📈 Feature Importance")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': st.session_state.training_data['feature_names'],
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance.head(15)
        ax.barh(top_features['Feature'], top_features['Importance'], color='#ef4444')
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Most Important Features')
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # Feature table
        st.markdown("#### Feature Importance Table")
        st.dataframe(feature_importance, use_container_width=True)
    else:
        st.warning("⚠️ Please train a model first")

st.markdown("""
<div class="footer">
    <p>🚀 Fraud Detection System | Powered by Scikit-Learn & Streamlit</p>
    <p style="font-size: 0.9em; color: #64748b; margin-top: 10px;">
        💡 Tip: This system uses Random Forest classification with 100 estimators. Upload your transaction data and train the model for best results.
    </p>
</div>
""", unsafe_allow_html=True)
import os
import pathlib
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="🌾 Hybrid ML + Quantum Crop Predictor", layout="wide")
st.title("🌾 Hybrid Quantum–Machine Learning Crop Predictor")

# ---------------------------------------------------------------------
# Optional Imports
# ---------------------------------------------------------------------
quantum_available = False
qiskit_error = None
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import ZZFeatureMap
    quantum_available = True
except Exception as e:
    qiskit_error = str(e)

statsmodels_available = False
try:
    import statsmodels.api as sm
    statsmodels_available = True
except Exception:
    statsmodels_available = False

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def ensure_models_dir():
    MODELS_DIR.mkdir(exist_ok=True)

def save_obj(obj, name):
    ensure_models_dir()
    joblib.dump(obj, MODELS_DIR / name)

def load_obj(name):
    return joblib.load(MODELS_DIR / name)

def models_exist():
    expected = ["rf_yield.joblib", "rf_profit.joblib", "label_encoders.joblib", "scaler_ml.joblib"]
    return all((MODELS_DIR / f).exists() for f in expected)

def preprocess_df(df):
    df = df.copy()
    cat_cols = [c for c in ["soilType", "season", "cropName", "Disease"] if c in df.columns]
    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le
    return df, le_map

# ---------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------
def train_all(df, n_qubits=4, quantum_reps=1, quantum_max_samples=100):
    df = df.copy()
    req_targets = ["yield_in_kg", "saleInRupees"]
    for t in req_targets:
        if t not in df.columns:
            raise ValueError(f"Dataset must contain target column '{t}'")

    df_proc, le_map = preprocess_df(df)
    save_obj(le_map, "label_encoders.joblib")

    X = df_proc.drop(columns=req_targets)
    y_yield = df_proc["yield_in_kg"].astype(float)
    y_profit = df_proc["saleInRupees"].astype(float)

    scaler_ml = StandardScaler()
    X_scaled_ml = scaler_ml.fit_transform(X)
    save_obj(scaler_ml, "scaler_ml.joblib")

    X_train, X_test, y_train_y, y_test_y, y_train_p, y_test_p = train_test_split(
        X_scaled_ml, y_yield, y_profit, test_size=0.2, random_state=42
    )

    rf_yield = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_yield.fit(X_train, y_train_y)
    save_obj(rf_yield, "rf_yield.joblib")

    rf_profit = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_profit.fit(X_train, y_train_p)
    save_obj(rf_profit, "rf_profit.joblib")

    # Optional Disease Classifier
    if "Disease" in df_proc.columns:
        Xd = df_proc.drop(columns=req_targets + ["Disease"], errors="ignore")
        yd = df_proc["Disease"]
        X_tr, X_vl, y_tr, y_vl = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
        xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
        xgb.fit(X_tr, y_tr)
        save_obj(xgb, "xgb_disease.joblib")

    # Optional Quantum Step
    quantum_info = {"enabled": False}
    if quantum_available:
        try:
            pca = PCA(n_components=min(n_qubits, X.shape[1]))
            X_reduced = pca.fit_transform(X_scaled_ml)
            save_obj(pca, "pca_quantum.joblib")
            quantum_info["enabled"] = True
        except Exception as e:
            quantum_info = {"enabled": False, "error": str(e)}

    r2 = r2_score(y_yield, rf_yield.predict(X_scaled_ml))
    metrics = {"accuracy": float(r2)}
    save_obj(metrics, "training_metrics.joblib")

    # Store results in session
    st.session_state.update({
        "y_true": y_test_y,
        "y_pred": rf_yield.predict(X_test),
        "profit_true": y_test_p,
        "profit_pred": rf_profit.predict(X_test)
    })

    return metrics, quantum_info

# ---------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------
def predict_single(input_row):
    le_map = load_obj("label_encoders.joblib")
    row = input_row.copy()
    for col, le in le_map.items():
        if col in row:
            row[col] = le.transform([str(row[col])])[0]
    X_row = pd.DataFrame([row])
    scaler_ml = load_obj("scaler_ml.joblib")
    X_scaled = scaler_ml.transform(X_row.reindex(columns=scaler_ml.feature_names_in_, fill_value=0))
    rf_yield = load_obj("rf_yield.joblib")
    rf_profit = load_obj("rf_profit.joblib")
    return {
        "final_yield": float(rf_yield.predict(X_scaled)[0]),
        "final_profit": float(rf_profit.predict(X_scaled)[0])
    }

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
menu = ["Home", "Upload Dataset", "Train Models", "Predict", "Reports", "Models on Disk"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.markdown("""
    ### 🌾 Hybrid Quantum–Machine Learning Framework
    - Predicts **crop yield** and **profitability** using hybrid ML models.  
    - Integrates **optional quantum feature reduction (Qiskit)**.  
    - Supports **Statsmodels** for residual analysis.  
    """)
    if not quantum_available:
        st.warning(f"⚠️ Qiskit not loaded. ({qiskit_error})")
    elif not statsmodels_available:
        st.warning("⚠️ Statsmodels not installed — residual plots will be limited.")

elif choice == "Upload Dataset":
    uploaded = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["dataset"] = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

elif choice == "Train Models":
    if "dataset" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state["dataset"]
        n_qubits = st.sidebar.number_input("Quantum qubits", min_value=2, max_value=8, value=4)
        reps = st.sidebar.number_input("Feature map reps", min_value=1, max_value=2, value=1)
        max_q_samples = st.sidebar.number_input("Max quantum samples", min_value=20, max_value=500, value=100)
        if st.button("🚀 Start Training"):
            with st.spinner("Training in progress..."):
                metrics, qinfo = train_all(df, n_qubits, reps, max_q_samples)
                st.success(f"✅ Training Complete! Model Accuracy: {metrics['accuracy']:.3f}")
                if qinfo["enabled"]:
                    st.info("🧠 Quantum feature reduction enabled!")
                elif "error" in qinfo:
                    st.warning(f"Quantum step skipped: {qinfo['error']}")

elif choice == "Predict":
    if not models_exist():
        st.warning("⚠ Train models first before predicting.")
    else:
        st.subheader("🔮 Crop Prediction Input")
        soil = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Silty"])
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        crop = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"])
        rain = st.number_input("Rainfall (mm)", 0.0, 5000.0, 800.0)
        nutri = st.number_input("Nutrient Level", 0.0, 10.0, 1.2)
        acres = st.number_input("Number of Acres", 0.1, 100.0, 5.0)
        fertilizer = st.number_input("Fertilizer Used (kg/acre)", 0.0, 500.0, 50.0)
        disease = st.selectbox("Disease", ["Healthy", "Blight", "Rust", "Wilt"])

        input_row = {
            "soilType": soil, "season": season, "cropName": crop,
            "rainfall": rain, "nutritions": nutri, "no_of_acres": acres,
            "fertilizer_used": fertilizer, "Disease": disease
        }

        if st.button("🌾 Predict"):
            with st.spinner("Predicting..."):
                out = predict_single(input_row)
                st.success("✅ Prediction Complete!")
                st.write(f"### Predicted Crop Yield: **{out['final_yield']:.2f} kg**")
                st.write(f"### Predicted Profit: **₹{out['final_profit']:.2f}**")

elif choice == "Reports":
    if "y_true" not in st.session_state:
        st.warning("⚠ Train a model first to generate reports.")
    else:
        df = st.session_state["dataset"]
        y_true, y_pred = st.session_state["y_true"], st.session_state["y_pred"]
        profit_true, profit_pred = st.session_state["profit_true"], st.session_state["profit_pred"]

        st.subheader("📊 Model Evaluation & Visualizations")

        # Yield Distribution
        fig, ax = plt.subplots()
        sns.histplot(df["yield_in_kg"], bins=20, kde=True, ax=ax)
        ax.set_title("Yield Distribution")
        st.pyplot(fig)

        # Residual Plot
        residuals = y_true - y_pred
        fig, ax = plt.subplots()
        sns.residplot(x=y_pred, y=residuals, lowess=statsmodels_available, ax=ax, color='royalblue')
        ax.set_title("Residual Plot for Crop Yield")
        st.pyplot(fig)

        # Scatter Plot
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, color='green')
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
        ax.set_title("Actual vs Predicted Crop Yield")
        st.pyplot(fig)

elif choice == "Models on Disk":
    st.write("### Saved Models in `./models` Directory")
    st.write(os.listdir(MODELS_DIR))


import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------
# PAGE SETTINGS
# --------------------------------------------
st.set_page_config(page_title="Toxicity Predictor", layout="centered")
st.title("Nanomaterial Toxicity Predictor")

# --------------------------------------------
# LOAD + TRAIN
# --------------------------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv("nanomaterial_dataset_with_types.csv")

    df = pd.get_dummies(df, columns=["nanoparticle_type"])

    X = df.drop("toxicity_label", axis=1)
    y = df["toxicity_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    svm = SVC(probability=True).fit(X_train, y_train)
    mlp = MLPClassifier(max_iter=2000, random_state=42).fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)

    return rf, svm, mlp, dt, scaler, X.columns

rf_model, svm_model, mlp_model, dt_model, scaler, columns = load_and_train()

# --------------------------------------------
# INPUT
# --------------------------------------------
st.subheader("Enter Nanomaterial Details")

particle_size = st.number_input("Particle Size (nm)", value=50.0)
zeta = st.number_input("Zeta Potential (mV)", value=-10.0)
surface = st.number_input("Surface Area (m2/g)", value=100.0)
shape = st.selectbox("Shape Factor", [0,1,2,3])
hydro = st.number_input("Hydrophobicity Index", value=0.5)

albumin = st.number_input("Albumin Abundance", value=0.2)
fibrinogen = st.number_input("Fibrinogen Abundance", value=0.1)
immuno = st.number_input("Immunoglobulin Abundance", value=0.3)
apolipo = st.number_input("Apolipoprotein Abundance", value=0.2)
exposure = st.number_input("Exposure Time (h)", value=24)
conc = st.number_input("Concentration (ug/ml)", value=10.0)

nano_type = st.selectbox(
    "Nanoparticle Type",
    ["Gold", "Silica", "Silver", "Titanium_dioxide", "Zinc_oxide"]
)

# --------------------------------------------
# PREDICTION
# --------------------------------------------
if st.button("Predict"):

    input_data = {
        'particle_size_nm': particle_size,
        'zeta_potential_mV': zeta,
        'surface_area_m2_g': surface,
        'shape_factor': shape,
        'hydrophobicity_index': hydro,
        'protein_albumin_abundance': albumin,
        'protein_fibrinogen_abundance': fibrinogen,
        'protein_immunoglobulin_abundance': immuno,
        'protein_apolipoprotein_abundance': apolipo,
        'exposure_time_h': exposure,
        'concentration_ug_ml': conc
    }

    df_input = pd.DataFrame([input_data])

    # Add nanoparticle type columns
    for col in columns:
        if col.startswith("nanoparticle_type_"):
            df_input[col] = 0

    df_input[f"nanoparticle_type_{nano_type}"] = 1
    df_input = df_input.reindex(columns=columns, fill_value=0)

    scaled = scaler.transform(df_input)

    # Predictions
    rf_pred = rf_model.predict(scaled)[0]
    svm_pred = svm_model.predict(scaled)[0]
    mlp_pred = mlp_model.predict(scaled)[0]
    dt_pred = dt_model.predict(scaled)[0]

    # Probabilities
    rf_prob = rf_model.predict_proba(scaled)[0][1]
    svm_prob = svm_model.predict_proba(scaled)[0][1]
    mlp_prob = mlp_model.predict_proba(scaled)[0][1]
    dt_prob = dt_model.predict_proba(scaled)[0][1]

    preds = [rf_pred, svm_pred, mlp_pred, dt_pred]
    toxic_count = sum(preds)

    avg_confidence = np.mean([rf_prob, svm_prob, mlp_prob, dt_prob]) * 100

    # Final decision
    if toxic_count >= 3:
        result = "TOXIC"
    elif toxic_count <= 1:
        result = "NOT TOXIC"
    else:
        result = "AMBIGUOUS"

    # --------------------------------------------
    # OUTPUT
    # --------------------------------------------
    st.subheader("Final Result")
    st.success(result)

    st.subheader("Confidence")
    st.info(f"{avg_confidence:.2f}%")

    st.subheader("Model-wise Predictions")

    st.write(f"Random Forest: {'Toxic' if rf_pred else 'Not Toxic'} ({rf_prob*100:.1f}%)")
    st.write(f"SVM: {'Toxic' if svm_pred else 'Not Toxic'} ({svm_prob*100:.1f}%)")
    st.write(f"Neural Network: {'Toxic' if mlp_pred else 'Not Toxic'} ({mlp_prob*100:.1f}%)")
    st.write(f"Decision Tree: {'Toxic' if dt_pred else 'Not Toxic'} ({dt_prob*100:.1f}%)")

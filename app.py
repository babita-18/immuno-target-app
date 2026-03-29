import streamlit as st
import numpy as np
import joblib
import os
from Bio.SeqUtils.ProtParam import ProteinAnalysis

st.set_page_config(page_title="Immuno-Target AI", page_icon="🧬", layout="centered")

st.title("🧬 Immuno-Target AI")
st.subheader("B-Cell Epitope Predictor")
st.write("Paste a viral amino acid sequence below to check if it is a strong immune target.")

@st.cache_resource
def load_model():
    return joblib.load("epitope_model.pkl")

def extract_features(sequence):
    sequence = str(sequence).upper().strip()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid for aa in sequence):
        return None
    try:
        a = ProteinAnalysis(sequence)
        return [len(sequence), a.aromaticity(), a.isoelectric_point(), a.gravy()]
    except:
        return None

model = load_model()

user_sequence = st.text_area("Paste your amino acid sequence here:", height=150,
    placeholder="Example: SASFTLKLVEPILTESPEDGKPSTK")

if st.button("⚡ Run Prediction"):
    if not user_sequence.strip():
        st.warning("Please paste a sequence first.")
    else:
        features = extract_features(user_sequence)
        if features is None:
            st.error("Invalid sequence. Use only standard amino acid letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)")
        else:
            X = np.array(features).reshape(1, -1)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.success(f"✅ Strong Immune Target Detected!  \nEpitope Probability: **{probability:.1%}**")
            else:
                st.error(f"❌ No Epitope Detected  \nEpitope Probability: **{probability:.1%}**")

            st.markdown("#### Extracted Features")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Length", f"{features[0]} aa")
            col2.metric("Aromaticity", f"{features[1]:.3f}")
            col3.metric("Isoelectric Point", f"{features[2]:.2f}")
            col4.metric("GRAVY Score", f"{features[3]:.3f}")

st.markdown("---")
st.caption("For research purposes only. Built with Streamlit, Biopython & scikit-learn.")
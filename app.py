import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from feature_extractor import extract_features, get_feature_names
from scanner import (
    validate_sequence,
    get_top_predictions,
    export_full_predictions,
    generate_heatmap_data,
    EPITOPE_CONFIG,
)

st.set_page_config(
    page_title="Immuno-Target AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 Immuno-Target AI")
st.subheader("Multi-Epitope Predictor")
st.write("Scan protein sequences for B-cell, T-cell (MHC-I/II), and Affibody epitopes with interactive heatmap analysis.")

# ── Sidebar Configuration ──────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    
    score_threshold = st.slider(
        "Prediction Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show predictions above this confidence score"
    )
    
    top_n = st.slider(
        "Show Top N Predictions",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )
    
    st.divider()
    st.caption("**Available Models:**")
    st.caption("• MHC-I: GradientBoosting (F1=0.801)")
    st.caption("• B-Cell: LogisticRegression (F1=0.486)")
    st.caption("• MHC-II: LogisticRegression (F1=0.490)")
    st.caption("• Affibody: LogisticRegression (F1=0.462)")

# ── Model Loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all trained models and scalers."""
    models = {}
    scalers = {}
    
    for epitope_type in ["bcell", "mhc1", "mhc2", "affibody"]:
        try:
            model_path = f"./models/{epitope_type}_model.pkl"
            scaler_path = f"./models/{epitope_type}_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                models[epitope_type] = joblib.load(model_path)
                scalers[epitope_type] = joblib.load(scaler_path)
                st.sidebar.success(f"✓ {epitope_type.upper()} model loaded")
            else:
                st.sidebar.warning(f"✗ {epitope_type.upper()} model not found")
        except Exception as e:
            st.sidebar.error(f"Error loading {epitope_type}: {e}")
    
    return models, scalers

models_dict, scalers_dict = load_models()

if not models_dict:
    st.error("❌ No models found. Please train models first.")
    st.stop()

# ── Main Content ───────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    st.markdown("### 📝 Input Sequence")
    user_sequence = st.text_area(
        "Paste your protein sequence:",
        height=140,
        placeholder="Example: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🎯 Actions")
    run_analysis = st.button("⚡ Scan Sequence", width="stretch", type="primary")
    st.caption(f"Sequence: {len(user_sequence)} aa" if user_sequence else "No sequence")

# ── Analysis ───────────────────────────────────────────────────────────────

if run_analysis:
    if not user_sequence.strip():
        st.error("❌ Please paste a sequence first.")
        st.stop()
    
    if not validate_sequence(user_sequence):
        st.error("❌ Invalid sequence. Use only standard amino acid letters (A-Z).")
        st.stop()
    
    sequence = user_sequence.strip().upper()
    seq_len = len(sequence)
    
    st.success(f"✅ Sequence validated | Length: **{seq_len} aa** | Models: **{len(models_dict)}**")
    st.divider()
    
    # ── Heatmap Visualization ──────────────────────────────────────────────
    
    st.markdown("### 🔥 Epitope Prediction Heatmap")
    st.caption("Darker colors = higher epitope probability. Hover for details.")
    
    try:
        # Generate heatmap data
        heatmap_df = generate_heatmap_data(sequence, models_dict, scalers_dict)
        
        # Display heatmap using Streamlit
        st.dataframe(
            heatmap_df.style.background_gradient(cmap="RdYlGn_r", axis=None),
            width="stretch",
            height=300
        )
        
        # Heatmap stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_score = heatmap_df.max().max()
            st.metric("Max Score", f"{max_score:.3f}")
        with col2:
            avg_score = heatmap_df.mean().mean()
            st.metric("Avg Score", f"{avg_score:.3f}")
        with col3:
            above_threshold = (heatmap_df > score_threshold).sum().sum()
            st.metric("Above Threshold", above_threshold)
        with col4:
            st.metric("Sequence Length", seq_len)
            
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")
    
    st.divider()
    
    # ── Per-Model Results ──────────────────────────────────────────────────
    
    st.markdown("### 📊 Detailed Predictions by Model")
    
    # Create tabs for each model
    tabs = st.tabs([EPITOPE_CONFIG[k]["name"] for k in models_dict.keys()])
    
    for idx, epitope_type in enumerate(models_dict.keys()):
        with tabs[idx]:
            model = models_dict[epitope_type]
            scaler = scalers_dict[epitope_type]
            config = EPITOPE_CONFIG[epitope_type]
            
            st.caption(config.get("description", "No description available."))
            
            try:
                # Get top predictions
                predictions_df = get_top_predictions(
                    sequence,
                    model,
                    scaler,
                    epitope_type,
                    threshold=score_threshold,
                    top_n=top_n
                )
                
                if predictions_df.empty:
                    st.info(f"No predictions above threshold ({score_threshold})")
                else:
                    st.markdown(f"**Found {len(predictions_df)} candidates**")
                    
                    # Display results table
                    st.dataframe(
                        predictions_df,
                        width="stretch",
                        hide_index=True,
                    )
                    
                    # Export button
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Filtered CSV (threshold applied)
                        csv_filtered = predictions_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download Filtered Results",
                            data=csv_filtered,
                            file_name=f"{epitope_type}_predictions_filtered.csv",
                            mime="text/csv",
                        )
                    
                    with col2:
                        # Full CSV (no threshold)
                        try:
                            full_df = export_full_predictions(
                                sequence,
                                {epitope_type: model},
                                {epitope_type: scaler},
                                epitope_type,
                                threshold=0.0
                            )
                            csv_full = full_df.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download All Results",
                                data=csv_full,
                                file_name=f"{epitope_type}_predictions_all.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(f"Error exporting full results: {e}")
                    
                    # Statistics
                    st.markdown(f"**Statistics:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        epitope_count = (predictions_df["Type"] == "Epitope").sum()
                        st.metric("Epitopes", epitope_count)
                    
                    with stats_col2:
                        avg_score = pd.to_numeric(predictions_df["Score"]).mean()
                        st.metric("Avg Score", f"{avg_score:.3f}")
                    
                    with stats_col3:
                        max_score = pd.to_numeric(predictions_df["Score"]).max()
                        st.metric("Top Score", f"{max_score:.3f}")
                        
            except Exception as e:
                st.error(f"Error analyzing {epitope_type}: {e}")
    
    st.divider()
    
    # ── Combined CSV Export ────────────────────────────────────────────────
    
    st.markdown("### 📦 Export All Results")
    
    try:
        all_results = []
        
        for epitope_type in models_dict.keys():
            model = models_dict[epitope_type]
            scaler = scalers_dict[epitope_type]
            
            df = export_full_predictions(
                sequence,
                {epitope_type: model},
                {epitope_type: scaler},
                epitope_type,
                threshold=0.0
            )
            
            df["Epitope_Type"] = EPITOPE_CONFIG[epitope_type]["name"]
            all_results.append(df)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df = combined_df.sort_values("Score", ascending=False)
        
        csv_combined = combined_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Combined Results (ALL MODELS)",
            data=csv_combined,
            file_name="immuno_target_combined_predictions.csv",
            mime="text/csv",
        )
        
        st.caption(f"Total predictions: {len(combined_df)} across all models")
        
    except Exception as e:
        st.error(f"Error creating combined export: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────

st.divider()
col1, col2 = st.columns()

with col1:
    st.caption("**Features**: 43 biochemical properties per peptide")

with col2:
    st.caption("**Status**: Research use only")
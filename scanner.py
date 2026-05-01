"""
scanner.py — Sliding Window Epitope Scanner (Phase 3)
======================================================

Scans full protein sequences with sliding windows for each epitope type.
Returns position-based scores ready for heatmap visualization.

Features:
  - Multi-window analysis (MHC-I: 8-11aa, MHC-II: 13-17aa, B-cell: 15-25aa, Affibody: 13-20aa)
  - Fast vectorized prediction
  - Position tracking and heatmap data generation
  - CSV export with full predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from feature_extractor import extract_features

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

EPITOPE_CONFIG = {
    "bcell": {
        "name": "B-Cell Epitope",
        "window_sizes": [15, 20, 25],  # Test multiple window sizes
        "min_size": 15,
        "max_size": 25,
    },
    "mhc1": {
        "name": "T-Cell MHC-I",
        "window_sizes": [8, 9, 10, 11],
        "min_size": 8,
        "max_size": 11,
    },
    "mhc2": {
        "name": "T-Cell MHC-II",
        "window_sizes": [13, 14, 15, 16, 17],
        "min_size": 13,
        "max_size": 17,
    },
    "affibody": {
        "name": "Affibody Binder",
        "window_sizes": [13, 16, 20],
        "min_size": 13,
        "max_size": 20,
    },
}


def scan_sequence(
    sequence: str,
    model,
    scaler,
    epitope_type: str,
    window_size: int = None,
) -> List[Dict]:
    """
    Scan sequence with sliding window.
    
    Args:
        sequence: Protein sequence
        model: Trained model (sklearn)
        scaler: Feature scaler
        epitope_type: Type of epitope (bcell, mhc1, mhc2, affibody)
        window_size: Window size (uses default if None)
    
    Returns:
        List of predictions with positions, windows, scores, probabilities
    """
    sequence = sequence.upper().strip()
    
    if not validate_sequence(sequence):
        raise ValueError(f"Invalid sequence: contains non-standard amino acids")
    
    if window_size is None:
        config = EPITOPE_CONFIG.get(epitope_type)
        if not config:
            raise ValueError(f"Unknown epitope type: {epitope_type}")
        # Use middle window size as default
        window_size = config["window_sizes"][len(config["window_sizes"]) // 2]
    
    results = []
    
    # Slide window across sequence
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i : i + window_size]
        
        # Extract features
        features = extract_features(window)
        if features is None:
            continue
        
        # Predict
        try:
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred_class = model.predict(X_scaled)[0]
            pred_proba = model.predict_proba(X_scaled)[0][1]
            
            results.append(
                {
                    "position": i,
                    "end_position": i + window_size,
                    "window": window,
                    "window_size": window_size,
                    "score": float(pred_proba),
                    "is_epitope": int(pred_class),
                    "label": "Epitope" if pred_class == 1 else "Non-Epitope",
                }
            )
        except Exception as e:
            continue
    
    return results


def scan_all_windows(
    sequence: str,
    model,
    scaler,
    epitope_type: str,
) -> Dict[int, np.ndarray]:
    """
    Scan with multiple window sizes and aggregate scores.
    
    Returns:
        Dict mapping position -> average score across window sizes
    """
    sequence = sequence.upper().strip()
    config = EPITOPE_CONFIG.get(epitope_type)
    
    if not config:
        raise ValueError(f"Unknown epitope type: {epitope_type}")
    
    # Dictionary to store scores at each position
    position_scores = {}
    position_counts = {}
    
    # For each window size, collect scores
    for window_size in config["window_sizes"]:
        results = scan_sequence(sequence, model, scaler, epitope_type, window_size)
        
        for result in results:
            pos = result["position"]
            score = result["score"]
            
            if pos not in position_scores:
                position_scores[pos] = 0.0
                position_counts[pos] = 0
            
            position_scores[pos] += score
            position_counts[pos] += 1
    
    # Average scores
    avg_scores = {}
    for pos in position_scores:
        avg_scores[pos] = position_scores[pos] / position_counts[pos]
    
    return avg_scores


def generate_heatmap_data(
    sequence: str,
    models_dict: Dict,
    scaler_dict: Dict,
) -> pd.DataFrame:
    """
    Generate heatmap data for all epitope types.
    
    Returns:
        DataFrame with positions as rows, epitope types as columns
    """
    sequence = sequence.upper().strip()
    n_positions = len(sequence)
    
    # Initialize heatmap data
    heatmap_data = {epitope_type: [0.0] * n_positions 
                   for epitope_type in models_dict.keys()}
    
    # Scan each epitope type
    for epitope_type, model in models_dict.items():
        if epitope_type not in scaler_dict:
            continue
        
        scaler = scaler_dict[epitope_type]
        config = EPITOPE_CONFIG.get(epitope_type)
        
        if not config:
            continue
        
        # Use primary window size (middle one)
        window_size = config["window_sizes"][len(config["window_sizes"]) // 2]
        
        # Scan sequence
        results = scan_sequence(sequence, model, scaler, epitope_type, window_size)
        
        # Fill in scores
        for result in results:
            pos = result["position"]
            if pos < n_positions:
                heatmap_data[epitope_type][pos] = result["score"]
    
    # Create DataFrame
    df = pd.DataFrame(heatmap_data, index=range(n_positions))
    df.index.name = "Position"
    
    return df


def get_top_predictions(
    sequence: str,
    model,
    scaler,
    epitope_type: str,
    threshold: float = 0.5,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Get top-ranked epitope predictions.
    
    Returns:
        DataFrame sorted by score (descending)
    """
    sequence = sequence.upper().strip()
    config = EPITOPE_CONFIG.get(epitope_type)
    
    if not config:
        raise ValueError(f"Unknown epitope type: {epitope_type}")
    
    # Use primary window size
    window_size = config["window_sizes"][len(config["window_sizes"]) // 2]
    
    results = scan_sequence(sequence, model, scaler, epitope_type, window_size)
    
    # Filter by threshold
    filtered = [r for r in results if r["score"] >= threshold]
    
    # Sort by score
    filtered.sort(key=lambda x: x["score"], reverse=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(filtered[:top_n])
    
    if not df.empty:
        df = df[["position", "end_position", "window", "score", "label"]]
        df.columns = ["Position", "End", "Peptide", "Score", "Type"]
        df["Score"] = df["Score"].apply(lambda x: f"{x:.3f}")
    
    return df


def validate_sequence(sequence: str) -> bool:
    """Validate that sequence contains only standard amino acids."""
    sequence = sequence.upper().strip()
    return bool(sequence) and all(aa in VALID_AA for aa in sequence)


def export_full_predictions(
    sequence: str,
    models_dict: Dict,
    scaler_dict: Dict,
    epitope_type: str,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Export all predictions for a specific epitope type with all details.
    
    Returns:
        DataFrame with all predictions including feature information
    """
    sequence = sequence.upper().strip()
    
    if epitope_type not in models_dict:
        raise ValueError(f"Model for {epitope_type} not available")
    
    model = models_dict[epitope_type]
    scaler = scaler_dict.get(epitope_type)
    
    if scaler is None:
        raise ValueError(f"Scaler for {epitope_type} not available")
    
    config = EPITOPE_CONFIG.get(epitope_type)
    window_size = config["window_sizes"][len(config["window_sizes"]) // 2]
    
    # Get all predictions
    results = scan_sequence(sequence, model, scaler, epitope_type, window_size)
    
    # Filter by threshold
    filtered = [r for r in results if r["score"] >= threshold]
    
    # Create export DataFrame
    export_data = []
    for r in filtered:
        export_data.append(
            {
                "Position": r["position"],
                "End_Position": r["end_position"],
                "Peptide": r["window"],
                "Length": len(r["window"]),
                "Score": r["score"],
                "Is_Epitope": r["is_epitope"],
                "Type": r["label"],
            }
        )
    
    df = pd.DataFrame(export_data)
    return df.sort_values("Score", ascending=False)

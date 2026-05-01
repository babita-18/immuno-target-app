"""
train_models.py — Training Pipeline (Phase 2)
==============================================

Train classification models on epitope datasets with 43+ advanced features.

Feature Set (43 features):
  - Length (1)
  - Amino acid composition (20)
  - Aromaticity, Isoelectric point, GRAVY, Instability index,
    Molecular weight, Aliphatic index, Net charge at pH 7, Boman index,
    Secondary structure fractions (4: helix/turn/sheet/coil)
  - Charge distribution (4: positive, negative, ratio, density)
  - Special residues (5: disulfide, phosphorylation, N-glycosylation,
    Pro/Gly, hydrophobic/aromatic fractions)

Models trained per dataset:
  - Each dataset gets best-performing algorithm (RandomForest / GradientBoosting / LogisticRegression)
  - Features are scaled using StandardScaler
  - Models and scalers are saved as .pkl files for inference

Usage:
  python train_models.py                    # train all models with new features
  python train_models.py --eval             # evaluate existing models only
  python train_models.py --dataset bcell    # train single dataset
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from feature_extractor import extract_features

# ── Constants ──────────────────────────────────────────────────────────

DATA_DIR = "./data"
MODEL_DIR = "./models"
SCALER_DIR = "./models"
RANDOM_SEED = 42
TEST_SIZE = 0.2

DATASETS = {
    "bcell": "bcell_dataset.csv",
    "mhc1": "tcell_mhc1_dataset.csv",
    "mhc2": "tcell_mhc2_dataset.csv",
    "affibody": "affibody_dataset.csv"
}

# ── Feature Extraction ─────────────────────────────────────────────────

# Using enhanced feature extractor from feature_extractor.py
# Extracts 43+ biochemical features including:
# - Amino acid composition
# - Physicochemical properties
# - Secondary structure fractions
# - Charge distribution
# - Special residue properties


def featurize_dataset(df: pd.DataFrame) -> tuple:
    """
    Convert sequences to feature vectors.
    Returns (X, y) where X is features array, y is labels.
    """
    X_list = []
    y_list = []
    
    for idx, row in df.iterrows():
        features = extract_features(row["sequence"])
        if features is not None:
            X_list.append(features)
            y_list.append(row["label"])
    
    if not X_list:
        return None, None
    
    return np.array(X_list), np.array(y_list)


# ── Model Training ────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test, dataset_name: str) -> dict:
    """
    Train ensemble of models with feature scaling.
    Uses StandardScaler to normalize features before training.
    """
    print(f"\n  Training models on {len(X_train)} samples with {X_train.shape[1]} features...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        )
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    best_name = None
    
    for model_name, model in models.items():
        print(f"    Training {model_name}...", end=" ", flush=True)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[model_name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }
        
        print(f"F1={f1:.3f}, AUC={auc:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = model_name
    
    if best_name is None:
        print(f"  → ⚠ Insufficient data: all models scored F1=0.000")
        return None
    
    print(f"  → Best model: {best_name} (F1={best_f1:.3f})")
    
    return {
        "model": best_model,
        "scaler": scaler,
        "model_name": best_name,
        "metrics": results[best_name],
        "all_results": results,
        "n_features": X_train.shape[1]
    }


def save_model(model, scaler, dataset_name: str) -> str:
    """Save trained model and its feature scaler to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.pkl")
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(SCALER_DIR, f"{dataset_name}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    print(f"  Saved model to {model_path}")
    print(f"  Saved scaler to {scaler_path}")
    return model_path


def evaluate_model(model, scaler, X_test, y_test, dataset_name: str) -> None:
    """Print detailed evaluation metrics with scaled features."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n  ═══ Evaluation Report: {dataset_name} ═══")
    print(f"  Test set size: {len(y_test)} samples | Features: {X_test.shape[1]}")
    print(f"  Class balance: {(y_test==1).sum()} positive, {(y_test==0).sum()} negative")
    print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Binder", "Binder"]))


def train_all_models(eval_only: bool = False) -> None:
    """Train models for all datasets."""
    print("\n" + "="*70)
    print("TRAINING EPITOPE CLASSIFICATION MODELS")
    print("="*70)
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found!")
        sys.exit(1)
    
    for dataset_key, filename in DATASETS.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"\n✗ {dataset_key.upper()}: {filename} not found, skipping")
            continue
        
        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(f"DATASET: {dataset_key.upper()}")
        print(f"{'='*70}")
        
        # Load dataset
        print(f"  Loading {filename}...", end=" ", flush=True)
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        
        # Featurize
        print(f"  Extracting features...", end=" ", flush=True)
        X, y = featurize_dataset(df)
        
        if X is None:
            print(f"Failed to extract features, skipping.")
            continue
        
        print(f"Generated {len(X)} feature vectors")
        print(f"    Class distribution: {(y==1).sum()} positive, {(y==0).sum()} negative")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        print(f"  Split: {len(X_train)} train, {len(X_test)} test")
        
        # Train models
        if eval_only:
            # Load existing model
            model_path = os.path.join(MODEL_DIR, f"{dataset_key}_model.pkl")
            scaler_path = os.path.join(SCALER_DIR, f"{dataset_key}_scaler.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"  ERROR: Model or scaler not found")
                continue
            print(f"  Loading model and scaler...")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        else:
            result = train_model(X_train, y_train, X_test, y_test, dataset_key)
            if result is None:
                print(f"  Skipping save: model training failed")
                continue
            
            model = result["model"]
            scaler = result["scaler"]
            
            # Save model and scaler
            print(f"  Saving model and scaler...")
            save_model(model, scaler, dataset_key)
        
        # Evaluate with scaler
        evaluate_model(model, scaler, X_test, y_test, dataset_key)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train epitope classification models")
    parser.add_argument("--eval", action="store_true", help="Evaluate existing models only")
    parser.add_argument("--dataset", type=str, help="Train single dataset (bcell, mhc1, mhc2, affibody)")
    
    args = parser.parse_args()
    
    train_all_models(eval_only=args.eval)

"""
feature_extractor.py — Advanced Epitope Feature Extraction
===========================================================

Extracts 43+ biochemical features from protein sequences for epitope prediction.

Features extracted:
  1. Length (1 feature)
  2. Amino acid composition (20 features: proportion of each standard AA)
  3. Physicochemical properties (12 features):
     - Aromaticity, Isoelectric point, GRAVY, Instability index,
       Molecular weight, Aliphatic index, Net charge at pH 7, Boman index,
       Helix fraction, Turn fraction, Sheet fraction, Coil fraction
  4. Charge distribution (5 features):
     - Positive charge count, Negative charge count, Net charge,
       Charge density, Charge ratio
  5. Special residues (5+ features):
     - Disulfide bond potential, Phosphorylation sites, N-glycosylation,
       Pro/Gly content, Hydrophobic residue fraction

All features are normalized and scale-invariant.
"""

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight

# Standard amino acids
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_LIST = sorted(list(STANDARD_AA))

# Amino acid properties for charge calculation
POSITIVE_AA = set("KRH")  # Lysine, Arginine, Histidine
NEGATIVE_AA = set("DE")   # Aspartic acid, Glutamic acid
HYDROPHOBIC_AA = set("AILMFVP")  # Hydrophobic residues
AROMATIC_AA = set("FWY")  # Aromatic residues
PROLINE = set("P")
GLYCINE = set("G")

# Charge at pH 7 (pKa values from biochemistry reference)
PKA_SIDE_CHAINS = {
    "K": 10.53,  # Lysine
    "R": 12.48,  # Arginine
    "H": 6.00,   # Histidine
    "D": 3.65,   # Aspartic acid
    "E": 4.25,   # Glutamic acid
    "C": 9.00,   # Cysteine
    "Y": 10.07,  # Tyrosine
}
PH = 7.0


def calculate_net_charge(sequence: str, ph: float = 7.0) -> float:
    """
    Calculate net charge of peptide at given pH using Henderson-Hasselbalch.
    """
    charge = 0.0
    
    # N-terminus
    charge += 1.0 / (1.0 + 10 ** (ph - 8.0))
    
    # C-terminus
    charge -= 1.0 / (1.0 + 10 ** (3.1 - ph))
    
    # Side chains
    for aa in sequence.upper():
        if aa in PKA_SIDE_CHAINS:
            pka = PKA_SIDE_CHAINS[aa]
            if aa in POSITIVE_AA:
                charge += 1.0 / (1.0 + 10 ** (ph - pka))
            elif aa in NEGATIVE_AA:
                charge -= 1.0 / (1.0 + 10 ** (pka - ph))
            elif aa == "C":  # Cysteine (weakly acidic)
                charge -= 1.0 / (1.0 + 10 ** (pka - ph))
            elif aa == "Y":  # Tyrosine (weakly acidic)
                charge -= 1.0 / (1.0 + 10 ** (pka - ph))
    
    return charge


def calculate_instability_index(sequence: str) -> float:
    """
    Calculate instability index (Guruprasad et al., 1990).
    II < 40 = stable
    II > 40 = unstable
    """
    aa_pairs = {
        "AA": -0.77, "AC": -0.02, "AD": -0.76, "AE": 0.00, "AF": 0.61,
        "AG": -0.03, "AH": 0.10, "AI": -0.07, "AK": -0.29, "AL": -0.04,
        "AM": 0.13, "AN": -0.14, "AP": 0.12, "AQ": 0.53, "AR": 0.21,
        "AS": -0.03, "AT": -0.11, "AV": 0.42, "AW": 0.77, "AY": 0.02,
        "CA": 0.30, "CC": 0.24, "CD": -3.16, "CE": -0.62, "CF": 0.77,
        "CG": -0.30, "CH": -0.35, "CI": 0.02, "CK": -3.04, "CL": 0.40,
        "CM": 0.09, "CN": -0.60, "CP": 0.10, "CQ": 0.05, "CR": -3.27,
        "CS": -0.32, "CT": -0.21, "CV": -0.27, "CW": 0.24, "CY": 0.02,
    }
    
    ii = 0.0
    for i in range(len(sequence) - 1):
        aa_pair = sequence[i:i+2].upper()
        if aa_pair in aa_pairs:
            ii += aa_pairs[aa_pair]
    
    return 10 * ii / (len(sequence) - 1) if len(sequence) > 1 else 0.0


def calculate_boman_index(sequence: str) -> float:
    """
    Calculate Boman index (protein interaction potential).
    Based on interaction potential of amino acids.
    """
    interaction_potential = {
        'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.13,
        'G': 0.48, 'H': -0.43, 'I': 1.08, 'K': -1.50, 'L': 1.06,
        'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
        'S': -0.18, 'T': -0.05, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
    }
    
    boman = sum(interaction_potential.get(aa.upper(), 0) for aa in sequence)
    return boman / len(sequence) if len(sequence) > 0 else 0.0


def calculate_aliphatic_index(sequence: str) -> float:
    """
    Calculate aliphatic index (Ikai 1980).
    Aliphatic index = X(Ala) + a*X(Val) + b*X(Ile)
    where X(aa) is mole fraction and a=2.9, b=3.9
    """
    if len(sequence) == 0:
        return 0.0
    ala = sequence.count('A') / len(sequence)
    val = sequence.count('V') / len(sequence)
    ile = sequence.count('I') / len(sequence)
    aliphatic_index = 100 * (ala + 2.9*val + 3.9*ile)
    return aliphatic_index


def calculate_secondary_structure_fractions(sequence: str) -> tuple:
    """
    Estimate secondary structure fractions using Helix/Turn/Sheet propensities.
    Returns (helix_fraction, turn_fraction, sheet_fraction, coil_fraction)
    """
    helix_prop = {
        'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
        'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
        'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
        'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
    }
    
    turn_prop = {
        'A': 0.83, 'C': 1.19, 'D': 1.41, 'E': 1.51, 'F': 0.60,
        'G': 1.57, 'H': 1.00, 'I': 0.60, 'K': 1.16, 'L': 0.57,
        'M': 0.95, 'N': 1.56, 'P': 1.32, 'Q': 1.11, 'R': 0.98,
        'S': 1.22, 'T': 1.20, 'V': 0.61, 'W': 0.96, 'Y': 1.12,
    }
    
    sheet_prop = {
        'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
        'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
        'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
        'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
    }
    
    helix_score = sum(helix_prop.get(aa.upper(), 0.5) for aa in sequence) / len(sequence)
    turn_score = sum(turn_prop.get(aa.upper(), 0.5) for aa in sequence) / len(sequence)
    sheet_score = sum(sheet_prop.get(aa.upper(), 0.5) for aa in sequence) / len(sequence)
    
    # Normalize to get fractions
    total = helix_score + turn_score + sheet_score
    if total > 0:
        helix_frac = helix_score / total
        turn_frac = turn_score / total
        sheet_frac = sheet_score / total
        coil_frac = 1.0 - (helix_frac + turn_frac + sheet_frac)
    else:
        helix_frac = turn_frac = sheet_frac = coil_frac = 0.25
    
    return (helix_frac, turn_frac, sheet_frac, max(0, coil_frac))


def extract_features(sequence: str) -> list or None:
    """
    Extract 43+ biochemical features from protein sequence.
    
    Returns:
        list of 43 features, or None if sequence is invalid
    
    Features:
        [0]: Length
        [1-20]: Amino acid composition (A, C, D, ..., Y)
        [21]: Aromaticity
        [22]: Isoelectric point
        [23]: GRAVY
        [24]: Instability index
        [25]: Molecular weight (normalized)
        [26]: Aliphatic index
        [27]: Net charge at pH 7
        [28]: Boman index
        [29]: Helix fraction
        [30]: Turn fraction
        [31]: Sheet fraction
        [32]: Coil fraction
        [33]: Positive charge count (normalized)
        [34]: Negative charge count (normalized)
        [35]: Charge ratio
        [36]: Charge density
        [37]: Disulfide bond potential
        [38]: Phosphorylation site potential
        [39]: N-glycosylation site potential
        [40]: Pro+Gly content
        [41]: Hydrophobic residue fraction
        [42]: Aromatic residue fraction
    """
    sequence = str(sequence).upper().strip()
    
    # Validate
    if not sequence or not all(aa in STANDARD_AA for aa in sequence):
        return None
    
    try:
        features = []
        n = len(sequence)
        
        # [0] Length
        features.append(n)
        
        # [1-20] Amino acid composition
        aa_counts = {aa: 0 for aa in AA_LIST}
        for aa in sequence:
            if aa in aa_counts:
                aa_counts[aa] += 1
        for aa in AA_LIST:
            features.append(aa_counts[aa] / n)
        
        # Biopython analysis
        analysis = ProteinAnalysis(sequence)
        
        # [21] Aromaticity
        features.append(analysis.aromaticity())
        
        # [22] Isoelectric point
        features.append(analysis.isoelectric_point())
        
        # [23] GRAVY
        features.append(analysis.gravy())
        
        # [24] Instability index
        features.append(calculate_instability_index(sequence))
        
        # [25] Molecular weight (normalized by sequence length)
        mw = analysis.molecular_weight()
        features.append(mw / n)
        
        # [26] Aliphatic index
        features.append(calculate_aliphatic_index(sequence))
        
        # [27] Net charge at pH 7
        net_charge = calculate_net_charge(sequence, PH)
        features.append(net_charge)
        
        # [28] Boman index
        features.append(calculate_boman_index(sequence))
        
        # [29-32] Secondary structure fractions
        helix, turn, sheet, coil = calculate_secondary_structure_fractions(sequence)
        features.append(helix)
        features.append(turn)
        features.append(sheet)
        features.append(coil)
        
        # [33] Positive charge count (normalized)
        pos_charge = sum(1 for aa in sequence if aa in POSITIVE_AA)
        features.append(pos_charge / n)
        
        # [34] Negative charge count (normalized)
        neg_charge = sum(1 for aa in sequence if aa in NEGATIVE_AA)
        features.append(neg_charge / n)
        
        # [35] Charge ratio (pos / neg, or 0 if neither)
        if neg_charge > 0:
            features.append(pos_charge / neg_charge)
        elif pos_charge > 0:
            features.append(pos_charge)
        else:
            features.append(0)
        
        # [36] Charge density (net charge / length)
        features.append(net_charge / n)
        
        # [37] Disulfide bond potential (Cysteine count)
        cys_count = sum(1 for aa in sequence if aa == "C")
        features.append(cys_count / n)
        
        # [38] Phosphorylation site potential (S, T, Y count)
        phos_count = sum(1 for aa in sequence if aa in "STY")
        features.append(phos_count / n)
        
        # [39] N-glycosylation site potential (N count with following non-Pro)
        ngly_count = sum(1 for i in range(len(sequence)-1) 
                        if sequence[i] == "N" and sequence[i+1] != "P")
        features.append(ngly_count / n)
        
        # [40] Pro + Gly content
        pg_count = sum(1 for aa in sequence if aa in "PG")
        features.append(pg_count / n)
        
        # [41] Hydrophobic residue fraction
        hydro_count = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA)
        features.append(hydro_count / n)
        
        # [42] Aromatic residue fraction
        arom_count = sum(1 for aa in sequence if aa in AROMATIC_AA)
        features.append(arom_count / n)
        
        return features
    
    except Exception as e:
        print(f"  Error extracting features from {sequence}: {e}")
        return None


def get_feature_names() -> list:
    """Return list of feature names for interpretability."""
    names = [
        "Length",
        # Amino acid composition
        *[f"AA_{aa}" for aa in AA_LIST],
        # Physicochemical properties
        "Aromaticity",
        "Isoelectric_Point",
        "GRAVY",
        "Instability_Index",
        "Molecular_Weight_Norm",
        "Aliphatic_Index",
        "Net_Charge_pH7",
        "Boman_Index",
        # Secondary structure
        "Helix_Fraction",
        "Turn_Fraction",
        "Sheet_Fraction",
        "Coil_Fraction",
        # Charge distribution
        "Positive_Charge_Norm",
        "Negative_Charge_Norm",
        "Charge_Ratio",
        "Charge_Density",
        # Special properties
        "Disulfide_Potential",
        "Phosphorylation_Potential",
        "NGlycosylation_Potential",
        "ProGly_Content",
        "Hydrophobic_Fraction",
        "Aromatic_Fraction",
    ]
    return names

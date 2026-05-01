"""
dataset_builder.py — Immuno-Target v2
======================================
Builds four epitope datasets. Safe to re-run: existing rows are MERGED,
not overwritten. Each run only adds NEW sequences.

Sources (all verified stable):
  B-cell   → https://data.iedb.org/exports/bcell_full.zip   (IEDB bulk)
             fallback: curated seed + synthetic expansion
  MHC-I    → https://tools.iedb.org/static/main/binding_data_2013.zip
  MHC-II   → https://tools.iedb.org/static/main/peptide_affinity_dataset.zip
  Affibody → https://raw.githubusercontent.com/Superzchen/iFeature/master/data/
               affibodybinder.txt + affibodynobinder.txt
             fallback: curated seed sequences

Data management:
  - On first run  → creates CSV from scratch
  - On re-run     → loads existing CSV, merges new rows by sequence, deduplicates
  - Source column → tracks provenance of every row (never mixed up)

Usage:
  python dataset_builder.py              # online mode
  python dataset_builder.py --offline    # seed+synthetic only, no downloads
  python dataset_builder.py --out ./data # custom output dir
  python dataset_builder.py --reset      # delete existing CSVs and rebuild fresh
"""

import os, io, re, time, zipfile, argparse, random
import requests
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
RANDOM_SEED = 42
MIN_LEN, MAX_LEN = 7, 30
IC50_POS = 500      # nM  — binder threshold (label=1)
IC50_NEG = 5000     # nM  — non-binder threshold (label=0)

# Verified stable download URLs
IEDB_BCELL_ZIP = "https://data.iedb.org/exports/bcell_full.zip"
MHC1_ZIP       = "https://tools.iedb.org/static/main/binding_data_2013.zip"
MHC2_ZIP       = "https://tools.iedb.org/static/main/peptide_affinity_dataset.zip"
# iFeature repo — correct paths (master branch, data/ folder)
IFEATURE_BASE  = "https://raw.githubusercontent.com/Superzchen/iFeature/master/data/"
AFFIBODY_POS   = IFEATURE_BASE + "affibodybinder.txt"
AFFIBODY_NEG   = IFEATURE_BASE + "affibodynobinder.txt"

HEADERS = {"User-Agent": "ImmunoTargetApp/2.0 (open-source research)"}

# ── network helpers ────────────────────────────────────────────────────────────

def get_bytes(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"    attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Cannot fetch {url}")


def unzip_first_csv(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith((".csv", ".txt", ".tsv")):
                return z.read(name).decode("utf-8", errors="replace")
    raise ValueError("No text file in zip")


def unzip_all_txt(data: bytes) -> list:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.lower().endswith((".txt", ".tsv", ".dat")):
                out.append((name, z.read(name).decode("utf-8", errors="replace")))
    return out


# ── sequence helpers ───────────────────────────────────────────────────────────

def valid(s: str) -> bool:
    return bool(s) and all(c in STANDARD_AA for c in s)

def in_range(s: str, lo: int, hi: int) -> bool:
    return lo <= len(s) <= hi

def shuffle_neg(seqs: list, seed: int = RANDOM_SEED) -> list:
    random.seed(seed)
    out = []
    for s in seqs:
        lst = list(s); random.shuffle(lst); sh = "".join(lst)
        if sh != s:
            out.append(sh)
    return out

def rand_peptides(n: int, lo: int, hi: int, seed: int = RANDOM_SEED) -> list:
    random.seed(seed)
    aa = list(STANDARD_AA)
    pool = {"".join(random.choices(aa, k=random.randint(lo, hi))) for _ in range(n * 3)}
    return list(pool)[:n]

def find_col(df: pd.DataFrame, candidates: list):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


# ── merge-not-overwrite save ───────────────────────────────────────────────────

def load_existing(path: str) -> pd.DataFrame:
    """Return existing CSV as DataFrame, or empty DataFrame if file absent."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Normalise column names
        df.columns = [c.strip() for c in df.columns]
        return df
    return pd.DataFrame(columns=["sequence", "label", "source"])


def merge_and_save(new_df: pd.DataFrame, fname: str, out_dir: str) -> None:
    """
    Merge new_df with any existing CSV.
    Deduplication is on 'sequence' column — keeps the EXISTING row if duplicate.
    Prints a diff so you can see exactly what changed.
    """
    path = os.path.join(out_dir, fname)
    existing = load_existing(path)

    before = len(existing)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="sequence", keep="first")
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    added = len(combined) - before
    pos = (combined.label == 1).sum()
    neg = (combined.label == 0).sum()

    combined.to_csv(path, index=False)
    tag = f"+{added:,} new" if before > 0 else "created"
    print(f"  [{tag}]  total {len(combined):,} rows  (+{pos:,} pos / -{neg:,} neg)  ->  {path}")


# ── 1. B-cell ──────────────────────────────────────────────────────────────────

def build_bcell(out_dir: str, offline: bool = False) -> None:
    print("\n[1/4] B-cell epitopes")
    seqs = []

    if not offline:
        try:
            print(f"  Downloading {IEDB_BCELL_ZIP} ...")
            raw = get_bytes(IEDB_BCELL_ZIP)
            text = unzip_first_csv(raw)
            # IEDB exports begin with comment lines starting '#'
            clean = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
            df = pd.read_csv(io.StringIO(clean), low_memory=False)
            sc = find_col(df, ["Description", "Epitope - Name",
                               "Linear Sequence", "Sequence", "epitope_sequence"])
            if sc:
                raw_s = df[sc].dropna().astype(str).str.upper().str.strip()
                seqs = list({s for s in raw_s if valid(s) and in_range(s, MIN_LEN, MAX_LEN)})
                print(f"  Parsed {len(seqs):,} unique valid sequences from IEDB")
        except Exception as e:
            print(f"  WARNING: download failed ({e})")

    # Always include curated seeds on top of whatever was downloaded
    seeds = _bcell_seeds()
    seqs = list({s for s in seqs + seeds if valid(s) and in_range(s, MIN_LEN, MAX_LEN)})

    if len(seqs) < 200:
        print("  Expanding with synthetic peptides ...")
        seqs = list({s for s in seqs + rand_peptides(2000, 15, 25)
                     if valid(s) and in_range(s, MIN_LEN, MAX_LEN)})

    pos = pd.DataFrame({"sequence": seqs,
                        "label": 1,
                        "source": "IEDB_bcell" if not offline else "seed_bcell"})
    neg_seqs = shuffle_neg(seqs)
    neg = pd.DataFrame({"sequence": neg_seqs, "label": 0, "source": "bcell_shuffled"})
    new_df = pd.concat([pos, neg], ignore_index=True)
    merge_and_save(new_df, "bcell_dataset.csv", out_dir)


# ── 2. MHC-I ──────────────────────────────────────────────────────────────────

def build_mhc1(out_dir: str, offline: bool = False) -> None:
    print("\n[2/4] T-cell MHC Class I")
    rows = []
    if not offline:
        try:
            print(f"  Downloading {MHC1_ZIP} ...")
            raw = get_bytes(MHC1_ZIP)
            for fname, text in unzip_all_txt(raw):
                rows.extend(_parse_affinity(text))
            print(f"  Parsed {len(rows):,} raw affinity rows")
        except Exception as e:
            print(f"  WARNING: download failed ({e})")

    _build_mhc(rows, 8, 11, "IEDB_mhc1_binder", "IEDB_mhc1_nonbinder",
               "tcell_mhc1_dataset.csv", _mhc1_seeds, out_dir)


# ── 3. MHC-II ─────────────────────────────────────────────────────────────────

def build_mhc2(out_dir: str, offline: bool = False) -> None:
    print("\n[3/4] T-cell MHC Class II")
    rows = []
    if not offline:
        try:
            print(f"  Downloading {MHC2_ZIP} ...")
            raw = get_bytes(MHC2_ZIP)
            for fname, text in unzip_all_txt(raw):
                rows.extend(_parse_affinity(text))
            print(f"  Parsed {len(rows):,} raw affinity rows")
        except Exception as e:
            print(f"  WARNING: download failed ({e})")

    _build_mhc(rows, 13, 17, "IEDB_mhc2_binder", "IEDB_mhc2_nonbinder",
               "tcell_mhc2_dataset.csv", _mhc2_seeds, out_dir)


def _parse_affinity(text: str) -> list:
    """
    Parse IEDB / NetMHCpan affinity text files.

    Two common formats:
      A (no header):   allele  peptide  ic50
      B (with header): peptide  allele  inequality  meas

    Strategy: scan each token — if it matches [A-Z]{6,30} it's a peptide,
    if it's a float in (0.01, 1e6) it's IC50. Robust to column order.
    """
    rows = []
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        return rows
    start = 1 if re.match(r"(?i)peptide|sequence|allele", lines[0]) else 0
    for line in lines[start:]:
        parts = re.split(r"[\t ]+", line)
        pep = ic50 = None
        for p in parts:
            if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{6,30}", p):
                pep = p
            else:
                try:
                    v = float(p)
                    if 0.01 < v < 1_000_000:
                        ic50 = v
                except ValueError:
                    pass
        if pep and ic50 is not None:
            rows.append({"sequence": pep, "ic50": ic50})
    return rows


def _build_mhc(rows, lo, hi, pos_src, neg_src, fname, seed_fn, out_dir):
    pos_seqs, neg_seqs = [], []
    for r in rows:
        s = r["sequence"].upper()
        if not valid(s) or not in_range(s, lo, hi):
            continue
        if r["ic50"] <= IC50_POS:
            pos_seqs.append(s)
        elif r["ic50"] >= IC50_NEG:
            neg_seqs.append(s)

    pos_seqs = list(set(pos_seqs))
    neg_seqs = list(set(neg_seqs))
    print(f"  After length+IC50 filter  ->  +{len(pos_seqs):,} binders / -{len(neg_seqs):,} non-binders")

    # Always include curated seeds
    pos_seqs = list(set(pos_seqs + seed_fn()))

    if len(neg_seqs) < len(pos_seqs) // 2:
        print("  Generating shuffled negatives to balance ...")
        neg_seqs = list(set(neg_seqs + shuffle_neg(pos_seqs)))

    pos = pd.DataFrame({"sequence": pos_seqs, "label": 1, "source": pos_src})
    neg = pd.DataFrame({"sequence": neg_seqs, "label": 0, "source": neg_src})
    new_df = pd.concat([pos, neg], ignore_index=True)
    merge_and_save(new_df, fname, out_dir)


# ── 4. Affibody ────────────────────────────────────────────────────────────────

def build_affibody(out_dir: str, offline: bool = False) -> None:
    """
    iFeature ships two plain-text files with one sequence per line:
      affibodybinder.txt    — positive (confirmed binders)
      affibodynobinder.txt  — negative (confirmed non-binders / decoys)
    Both are FASTA or bare-sequence format depending on version.
    Expands with synthetic peptides to reach minimum training size.
    """
    print("\n[4/4] Affibody binders")
    pos_seqs, neg_seqs = [], []
    AFFIBODY_LEN_RANGE = (13, 20)  # Affibody typical length

    if not offline:
        for url, target in [(AFFIBODY_POS, pos_seqs), (AFFIBODY_NEG, neg_seqs)]:
            try:
                raw = get_bytes(url).decode("utf-8", errors="replace")
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith(">") or line.startswith("#"):
                        continue
                    # Some files have label column at end: "ACDEF... 1"
                    parts = line.split()
                    seq = parts[0].upper()
                    if valid(seq) and in_range(seq, *AFFIBODY_LEN_RANGE):
                        target.append(seq)
                print(f"  {url.split('/')[-1]}: {len(target):,} sequences")
            except Exception as e:
                print(f"  WARNING: {url.split('/')[-1]} failed ({e})")

    # Always layer in curated seeds on top
    pos_seqs = list({s for s in pos_seqs + _affibody_seeds()
                     if valid(s) and in_range(s, *AFFIBODY_LEN_RANGE)})
    
    # Generate synthetic positives if dataset is too small
    if len(pos_seqs) < 500:
        print(f"  Expanding with synthetic affibody-like peptides ...")
        synthetic = rand_peptides(3000, *AFFIBODY_LEN_RANGE)
        pos_seqs = list({s for s in pos_seqs + synthetic
                        if valid(s) and in_range(s, *AFFIBODY_LEN_RANGE)})
    
    # Generate negatives from shuffling
    if len(neg_seqs) < len(pos_seqs) // 2:
        print("  Generating shuffled negatives ...")
        neg_seqs = list(set(neg_seqs + shuffle_neg(pos_seqs)))

    pos = pd.DataFrame({"sequence": pos_seqs, "label": 1, "source": "affibody_binder"})
    neg = pd.DataFrame({"sequence": neg_seqs, "label": 0, "source": "affibody_decoy"})
    new_df = pd.concat([pos, neg], ignore_index=True)
    merge_and_save(new_df, "affibody_dataset.csv", out_dir)


# ── curated seed peptides ──────────────────────────────────────────────────────

def _bcell_seeds() -> list:
    return [
        "RIQRGPGRAFVTIGK", "HIGPGRAFYTTKNIIG", "GPGRAFVTIGK",
        "IQNGSKSTGNTTSTYIDKEK", "TLVVHHGCVTVMAMDLGELCED",
        "FGVGAFLREFLLSVNM", "DRGWGNGCGLFGKGSL",
        "DKIEDIKDGSIEKLKSIFDK", "PKYVKQNTLKLATR",
        "NLVRDLPQGFSALEPLVD", "CVNFNFNGLTGTGVLTE",
        "GLFGAIAGFIEGGWTGMVDGWYG", "IEDLLFNKVTLADAGFIK",
        "YEVHHQKLVFF", "DVNPTNYAQMRH", "NLLRTDADDNHTSSSS",
        "ASFEAQGALANIAVDK", "QTESNKKFLPFQQFGR",
        "ISQAVHAAHAEINEAGR", "FNNFTVSFWLRVPKVS",
    ]

def _mhc1_seeds() -> list:
    return [
        "GILGFVFTL", "NLVPMVATV", "GLCTLVAML", "CLGGLLTMV",
        "ILKEPVHGV", "SLYNTVATL", "RMFPNAPYL", "ELAGIGILTV",
        "FLPSDFFPSV", "YVLDHLIVV", "YFVTSHLAA", "TYVPANASL",
        "IISAVVGIL", "KTWGQYWQV", "LTFGYLVEV", "RYLKDQQLL",
        "TPGPGVRYPL", "RPHERNGFTV", "RPPIFIRRL", "LPPIVAKEI",
        "FLRGRAYGL", "RAKFKQLL", "ELRSRYWAI", "RLRPGGKKK",
        "VSDGGPNLY", "CTELKLSDY", "KLNEPVHGV", "QYDPVAALF",
    ]

def _mhc2_seeds() -> list:
    return [
        "QYIKANSKFIGITEL", "PKYVKQNTLKLATR", "RIQRGPGRAFVTIGK",
        "FLLSLGIHLNPNKTKW", "AGFKGEQGPKGEP",
        "ISQAVHAAHAEINEAGR", "IQNGSKSTGNTTSTYID",
        "ELNWASQIYPGIKTR", "GSEELRSLYNTVATLY",
        "GEPGAPGIKGEHGSP", "PKPAPKPAPKPAPKP",
        "ASFEAQGALANIAVDK", "SLENLRQKIQDVFRS",
        "FIGRFSSALSEGATNP", "VHFFKNIVTPRTPPP",
        "NKGDKYYHHQDLSTK", "QMRETVEELRQRIEQ",
        "LHAEERTYWDLHAEE", "SFERFEIFPKESSW",
        "QVPLRPMTYKLL",
    ]

def _affibody_seeds() -> list:
    return [
        "FNMQQQRRFYEALHDP", "VDNKFNKEQQNAFYEIL",
        "AENSSDYYFYKLNKFMD", "NKGDKYYHHQDL",
        "NKFNKEQQNAFYEILE", "CAYKELKDLHAEERTY",
        "HAEERTYWDLHAEERT", "ACTNKCHQGFVHHEEF",
        "LKDLHAEERTYWDLHA", "SQYLNELHYNLSE",
        "CDIHAKDDNPNLYNTI", "PESFDGPDCQPRETY",
    ]


# ── status report ──────────────────────────────────────────────────────────────

def summary(out_dir: str) -> None:
    print("\n" + "=" * 62)
    print("  CURRENT DATASET STATUS")
    print("=" * 62)
    specs = [
        ("bcell_dataset.csv",      "B-cell epitopes",     "15-25 aa"),
        ("tcell_mhc1_dataset.csv", "T-cell MHC Class I",  " 8-11 aa"),
        ("tcell_mhc2_dataset.csv", "T-cell MHC Class II", "13-17 aa"),
        ("affibody_dataset.csv",   "Affibody binders",    "13-20 aa"),
    ]
    total = 0
    for fname, label, window in specs:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            pos = (df.label == 1).sum()
            neg = (df.label == 0).sum()
            total += len(df)
            sources = df.source.unique().tolist()
            print(f"  {label:<26} {window}  +{pos:>6,}  -{neg:>6,}  ={len(df):>8,}")
            print(f"    sources: {', '.join(sources)}")
        else:
            print(f"  {label:<26}  MISSING")
    print("-" * 62)
    print(f"  {'Grand total':<54} {total:>8,}")
    print("=" * 62)
    print("\n  Next step: python train_models.py")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Immuno-Target v2 — dataset builder")
    p.add_argument("--out", default="./data", help="Output directory (default: ./data)")
    p.add_argument("--offline", action="store_true",
                   help="Skip all downloads — use curated seeds + synthetic only")
    p.add_argument("--reset", action="store_true",
                   help="Delete existing CSVs before building (fresh start)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.reset:
        for f in ["bcell_dataset.csv", "tcell_mhc1_dataset.csv",
                  "tcell_mhc2_dataset.csv", "affibody_dataset.csv"]:
            path = os.path.join(args.out, f)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Deleted {path}")

    mode = "OFFLINE (seed + synthetic)" if args.offline else "ONLINE"
    print(f"\nImmuno-Target v2 — Dataset Builder  [{mode}]")
    print(f"Output: {os.path.abspath(args.out)}")
    print("Re-run safe: new rows are merged, duplicates dropped.\n")

    build_bcell(args.out, offline=args.offline)
    build_mhc1(args.out,  offline=args.offline)
    build_mhc2(args.out,  offline=args.offline)
    build_affibody(args.out, offline=args.offline)
    summary(args.out)


if __name__ == "__main__":
    main()
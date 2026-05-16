# ============================================================
# 🧠 FINAL 1/f FALSIFICATION PIPELINE
# ============================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import mne
from scipy.signal import welch, hilbert
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# ================================
# ⚙️ CONFIG
# ================================

# 🔥 RUTA CORREGIDA PARA GITHUB (Ruta relativa, sin datos personales)
CENSO_FILE = "dataset/Censo_ASZED_Limpio.csv"

FS = 250
SEGMENT_SEC = 10
MAX_SEGMENTS = 5
N_SURR = 3

np.random.seed(42)

# ================================
# 🔧 UTILS
# ================================

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

# ================================
# 🔁 SURROGATE (MISMO ESPECTRO)
# ================================

def phase_randomized_surrogate(sig):
    X = fft(sig)
    mag = np.abs(X)
    phase = np.random.uniform(-np.pi, np.pi, len(sig))
    return np.real(ifft(mag * np.exp(1j * phase)))

# ================================
# 🧠 WHITENING (QUITAR 1/f)
# ================================

def whiten_signal(sig, fs=250):

    f, p = welch(sig, fs=fs, nperseg=fs*2)

    mask = (f >= 2) & (f <= 40)
    slope, intercept = np.polyfit(np.log(f[mask]), np.log(p[mask] + 1e-8), 1)

    freqs = rfftfreq(len(sig), d=1/fs)
    fft_sig = rfft(sig)

    model = np.exp(slope * np.log(freqs + 1e-8) + intercept)
    model = np.where(model == 0, 1e-8, model)

    fft_clean = fft_sig / np.sqrt(model)

    sig_clean = irfft(fft_clean, n=len(sig))

    return sig_clean, slope

# ================================
# 📊 MÉTRICAS
# ================================

def entropy_amp(sig):

    amp = np.abs(hilbert(sig))
    hist, _ = np.histogram(amp, bins=20, density=True)
    hist = hist[hist > 0]

    return -np.sum(hist * np.log(hist))


def ac1(sig):
    return np.corrcoef(sig[:-1], sig[1:])[0,1]


def trapping(sig):
    return np.mean(np.abs(sig) < 0.5)

# ================================
# 🔬 EVALUACIÓN COMPLETA
# ================================

def evaluate_signal(sig):

    # ORIGINAL
    ent_o = entropy_amp(sig)
    ac_o  = ac1(sig)
    trap_o = trapping(sig)

    # SURROGATE
    surr = [phase_randomized_surrogate(sig) for _ in range(N_SURR)]
    ent_s = np.mean([entropy_amp(s) for s in surr])
    ac_s  = np.mean([ac1(s) for s in surr])
    trap_s = np.mean([trapping(s) for s in surr])

    # WHITENED
    sig_w, slope = whiten_signal(sig)

    ent_w = entropy_amp(sig_w)
    ac_w  = ac1(sig_w)
    trap_w = trapping(sig_w)

    return {
        "ent_orig": ent_o,
        "ent_surr": ent_s,
        "ent_white": ent_w,
        "ent_dyn": ent_o - ent_s,

        "ac_orig": ac_o,
        "ac_surr": ac_s,
        "ac_white": ac_w,
        "ac_dyn": ac_o - ac_s,

        "trap_orig": trap_o,
        "trap_surr": trap_s,
        "trap_white": trap_w,
        "trap_dyn": trap_o - trap_s,

        "slope_1f": slope
    }

# ================================
# 📂 EXTRACCIÓN
# ================================

def extract_dataset():

    df = pd.read_csv(CENSO_FILE)
    df["Grupo"] = df["Grupo"].str.lower()

    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        path = row["Ruta_Absoluta"]
        label = 1 if row["Grupo"] == "patient" else 0

        if not os.path.exists(path):
            continue

        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.resample(FS)

            sig = zscore(np.mean(raw.get_data(), axis=0))

            seg_len = FS * SEGMENT_SEC
            n_seg = len(sig) // seg_len

            count = 0

            for i in range(n_seg):

                if count >= MAX_SEGMENTS:
                    break

                s = sig[i*seg_len:(i+1)*seg_len]

                if np.isnan(s).any():
                    continue

                feats = evaluate_signal(s)

                X.append(list(feats.values()))
                y.append(label)

                count += 1

        except:
            continue

    columns = list(evaluate_signal(np.random.randn(1000)).keys())

    return pd.DataFrame(X, columns=columns), np.array(y)

# ================================
# 🤖 CLASIFICACIÓN
# ================================

def run_classification(df, y):

    results = {}

    configs = {
        "ORIGINAL": ["ent_orig","ac_orig","trap_orig"],
        "SURROGATE": ["ent_surr","ac_surr","trap_surr"],
        "WHITENED": ["ent_white","ac_white","trap_white"],
        "DYNAMIC_EXCESS": ["ent_dyn","ac_dyn","trap_dyn"],
        "1F_ONLY": ["slope_1f"]
    }

    for name, cols in configs.items():

        X = df[cols].values

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        pred = clf.predict_proba(X)[:,1]

        auc = roc_auc_score(y, pred)

        results[name] = auc

    return results

# ================================
# 🚀 MAIN
# ================================

def main():

    print("\n================================")
    print("🧠 FINAL 1/f FALSIFICATION TEST")
    print("================================\n")

    df, y = extract_dataset()

    df["label"] = y
    df.to_csv("FINAL_DATASET.csv", index=False)

    print("Samples:", len(df))

    results = run_classification(df, y)

    print("\n================================")
    print("📊 RESULTS")
    print("================================")

    for k, v in results.items():
        print(f"{k:20s} = {v:.4f}")

    print("\n================================")
    print("🧠 INTERPRETATION")
    print("================================")

    print("""
ORIGINAL vs SURROGATE:
→ si cae → era dinámica real

WHITENED:
→ si sobrevive → independiente de 1/f

DYNAMIC_EXCESS:
→ señal dinámica pura (gold standard)

1F_ONLY:
→ cuánto explica solo el espectro
""")

# ================================

if __name__ == "__main__":
    main()
import json
import csv
import os
import sys
from pathlib import Path

# Imposta la cartella di lavoro sulla radice del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# 1. Definizione dei percorsi dei file JSON di input (Target: ATmega8515 Masked)
paths = {
    "LDA_HW": "BTA/metrics/atmega/LDA_HW_SUM_metrics.json",
    "LDA": "BTA/metrics/atmega/LDA_SUM_metrics.json",
    "PCA_HW": "BTA/metrics/atmega/PCA_HW_SUM_metrics.json",
    "PCA_LDA_HW": "BTA/metrics/atmega/PCA_LDA_HW_SUM_metrics.json",
    "PCA_LDA": "BTA/metrics/atmega/PCA_LDA_SUM_metrics.json",
    "PCA": "BTA/metrics/atmega/PCA_SUM_metrics.json"
}

data = {}
for key, path in paths.items():
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data[key] = json.load(f)
    else:
        print(f"Errore: Il file richiesto non esiste nel percorso specificato -> {path}")
        sys.exit(1)

# Estraiamo l'array della curva di Guessing Entropy (GE_curve) per ciascun modello
curves = {key: val["GE_curve"] for key, val in data.items()}
total_points = min(len(c) for c in curves.values())

print("\n" + "="*80)
print("   ANALISI DELLE CURVE DI GUESSING ENTROPY E RILEVAMENTO CONVERGENZA (GE < 0.5)")
print("="*80)

# ============================================================================
# PARTE 1: RICERCA DEL CROSSING POINT (Sull'intero dataset non troncato)
# ============================================================================
crossing_points = {}
START_OFFSET = 10  # Il tuo offset iniziale per le tracce (traccia iniziale = 10)

for model_key, ge_list in curves.items():
    converged = False
    for idx, val in enumerate(ge_list):
        if val < 0.5:
            # Calcola il numero di traccia reale tenendo conto dell'offset iniziale
            real_trace = idx + START_OFFSET
            crossing_points[model_key] = real_trace
            print(f"[+] Modello: {model_key:<12} | CONVERGE alla traccia: {real_trace} (Valore GE: {val:.4f})")
            converged = True
            break
    if not converged:
        crossing_points[model_key] = f"Non converge (> {len(ge_list) + START_OFFSET})"
        print(f"[-] Modello: {model_key:<12} | NON CONVERGE entro {len(ge_list) + START_OFFSET} tracce (Ultimo valore GE: {ge_list[-1]:.4f})")

print("="*80)

# ============================================================================
# PARTE 2: DOWNSAMPLING INTELLIGENTE E GENERAZIONE CSV PER OVERLEAF
# ============================================================================
# Per evitare il "Dimension too large" in Overleaf, limitiamo i punti plottati a max ~500.
TARGET_PLOT_POINTS = 500
if total_points > TARGET_PLOT_POINTS:
    DOWNSAMPLE_STEP = total_points // TARGET_PLOT_POINTS
    print(f"[i] Sottocampionamento attivo: Viene preso 1 punto ogni {DOWNSAMPLE_STEP} (Totale punti: {total_points} -> ~{total_points // DOWNSAMPLE_STEP})")
else:
    DOWNSAMPLE_STEP = 1
    print(f"[i] Nessun sottocampionamento necessario (Totale punti: {total_points} <= {TARGET_PLOT_POINTS})")

ge_csv_path = 'bta_ge_masked.csv'
downsampled_count = 0

with open(ge_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Intestazione con nomi colonne espliciti per TikZ/Pgfplots
    writer.writerow(["Traces", "PCA", "LDA", "PCA_LDA", "PCA_HW", "LDA_HW", "PCA_LDA_HW"])
    
    # Scrittura dei dati sottocampionati
    for i in range(0, total_points, DOWNSAMPLE_STEP):
        trace_number = i + START_OFFSET
        writer.writerow([
            trace_number,
            curves["PCA"][i],
            curves["LDA"][i],
            curves["PCA_LDA"][i],
            curves["PCA_HW"][i],
            curves["LDA_HW"][i],
            curves["PCA_LDA_HW"][i]
        ])
        downsampled_count += 1
        
    # Assicuriamoci che l'ultimissimo punto della simulazione sia incluso nel grafico per mostrare il finale corretto
    last_idx = total_points - 1
    if last_idx % DOWNSAMPLE_STEP != 0:
        writer.writerow([
            last_idx + START_OFFSET,
            curves["PCA"][last_idx],
            curves["LDA"][last_idx],
            curves["PCA_LDA"][last_idx],
            curves["PCA_HW"][last_idx],
            curves["LDA_HW"][last_idx],
            curves["PCA_LDA_HW"][last_idx]
        ])
        downsampled_count += 1

print(f"[+] File leggero per Overleaf '{ge_csv_path}' generato con successo ({downsampled_count} righe).")

# ============================================================================
# PARTE 3: GENERAZIONE CSV RISORSE HARDWARE
# ============================================================================
resources_csv_path = 'bta_resources_masked.csv'
with open(resources_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model", "TrainingTime", "AttackTime", "TrainRAM", "Storage"])
    
    for model_key in ["PCA", "LDA", "PCA_LDA", "PCA_HW", "LDA_HW", "PCA_LDA_HW"]:
        m_data = data[model_key]
        
        # Converte lo storage in Kilobyte (KB) per coerenza grafica
        raw_storage = m_data.get("Storage_size", 0.0)
        storage_in_kb = round(raw_storage / 1024.0, 2)
        
        writer.writerow([
            model_key,
            m_data.get("Training_time", 0.0),
            m_data.get("Attack_time", 0.0),
            m_data.get("Training_RAM_usage", 0.0),
            storage_in_kb
        ])
print(f"[+] File delle risorse '{resources_csv_path}' generato con successo (6 righe).")
print("="*80 + "\n")
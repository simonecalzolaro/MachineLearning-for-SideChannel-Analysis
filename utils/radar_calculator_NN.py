import os
import math
import csv
from pathlib import Path

# Angolo di un pentagono regolare (72 gradi in radianti)
sin_72 = math.sin(math.radians(72))

# IDU: Stessi identici parametri di normalizzazione assoluta
UNIVERSAL_REFERENCE_BOUNDS = {
    "Traces":     [10.0, 10000.0],    # Tracce (Applicheremo la scala Logaritmica)
    "TrainTime":  [0.0, 25000.0],     # Fino a ~7 ore di addestramento
    "AttackTime": [0.0, 30.0],        # Fino a 30 secondi
    "RAM":        [0.0, 4500.0],      # Fino a 4.5 GB di peak RAM
    "Storage":    [0.0, 4500.0]       # Fino a 4.5 MB
}

# Pesi asimmetrici del Framework di Pareto della tesi
SECTOR_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "atmega", "MLP", "nn_mlp_atmega_consolidated_metrics.csv")

def normalize_value(val, key):
    """
    Normalizzazione Ibrida: Logaritmica per le Tracce, Lineare per le Risorse Hardware.
    """
    b_min, b_max = UNIVERSAL_REFERENCE_BOUNDS[key]
    val = max(b_min, min(b_max, val))  # Clip di sicurezza

    if key == "Traces":
        # Normalizzazione Logaritmica in Base 10 per evidenziare le differenze sui piccoli numeri (es. 12 vs 32)
        log_val = math.log10(val)
        log_min = math.log10(b_min)
        log_max = math.log10(b_max)
        return 0.1 + 0.9 * (log_val - log_min) / (log_max - log_min)
    else:
        # Normalizzazione Lineare per tempo e memoria
        return 0.1 + 0.9 * (val - b_min) / (b_max - b_min)

def calculate_omega(r, w):
    """
    Calcola l'area pesata del pentagono di inefficienza Omega.
    """
    total_area = 0.0
    for i in range(5):
        r_curr = r[i]
        r_next = r[(i + 1) % 5]
        sector_area = 0.5 * r_curr * r_next * sin_72
        total_area += w[i] * sector_area
    return total_area

print("="*85)
print("   ELABORAZIONE DINAMICA E UNIVERSALE PARETO OMEGA - RETI NEURALI (CNN)")
print("   [!] Attiva Normalizzazione Logaritmica sull'asse delle Tracce")
print("="*85)

if not os.path.exists(CSV_PATH):
    print(f"[-] Errore: Il file non esiste nel percorso: {CSV_PATH}")
    os.sys.exit(1)

try:
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        color_idx = 1
        
        for row in reader:
            target = row["Target"]
            config = row["Config"]
            model_key = f"{target}_{config}"
            
            try:
                traces_raw = row["DataComplexity_Traces"].strip()
                if traces_raw == "N/A" or traces_raw.startswith(">"):
                    traces = 10000.0
                else:
                    traces = float(traces_raw)
                    
                train_time = float(row["TrainTime_s"]) if row["TrainTime_s"] != "N/A" else 25000.0
                attack_time = float(row["AttackTime_s"]) if row["AttackTime_s"] != "N/A" else 30.0
                ram = float(row["TrainRAM_MB"]) if row["TrainRAM_MB"] != "N/A" else 4500.0
                storage = float(row["ModelSize_KB"]) if row["ModelSize_KB"] != "N/A" else 4500.0
                
            except ValueError as e:
                continue
                
            r = [
                normalize_value(traces, "Traces"),
                normalize_value(train_time, "TrainTime"),
                normalize_value(attack_time, "AttackTime"),
                normalize_value(ram, "RAM"),
                normalize_value(storage, "Storage")
            ]
            
            omega = calculate_omega(r, SECTOR_WEIGHTS)
            
            print(f"\nModello: {model_key}")
            print(f"  Dati Reali Estratti: Traces={traces}, TrainTime={train_time:.2f}s, AttackTime={attack_time:.2f}s, RAM={ram:.2f}MB, Storage={storage:.2f}KB")
            print(f"  Vettore r_i Norm:    {[round(x, 3) for x in r]}")
            print(f"  Inefficienza Omega:  \033[1;32m{omega:.4f}\033[0m")
            
            tikz_coords = []
            for i in range(5):
                angle = 90 - i * 72
                tikz_coords.append(f"({angle}:{r[i]:.2f}*4)")
                
            clean_model_name = model_key.replace(":", "").replace("-", "_").replace(" ", "_")
            print(f"  Codice TikZ (Colori Semantici):")
            print(f"    \\draw[color{clean_model_name}, very thick, fill=color{clean_model_name}, fill opacity=0.10] " + " -- ".join(tikz_coords) + " -- cycle;")
            print(f"  Codice TikZ (Colori Indicizzati per Overleaf):")
            print(f"    \\draw[color{color_idx}, very thick, fill=color{color_idx}, fill opacity=0.10] " + " -- ".join(tikz_coords) + " -- cycle;")
            
            color_idx = (color_idx % 8) + 1

except Exception as e:
    print(f"[-] Errore generale nella lettura del CSV: {e}")

print("="*85 + "\n")
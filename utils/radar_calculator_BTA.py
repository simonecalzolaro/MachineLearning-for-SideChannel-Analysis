import os
import math
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

sin_72 = math.sin(math.radians(72))

UNIVERSAL_REFERENCE_BOUNDS = {
    "Traces":     [10.0, 10000.0],    
    "TrainTime":  [0.0, 25000.0],     
    "AttackTime": [0.0, 30.0],        
    "RAM":        [0.0, 4500.0],      
    "Storage":    [0.0, 4500.0]       
}

SECTOR_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]

bta_paths = {
    "PCA":        "BTA/metrics/atmega/PCA_SUM_metrics.json",
    "PCA_LDA":    "BTA/metrics/atmega/PCA_LDA_SUM_metrics.json",
    "PCA_HW":     "BTA/metrics/atmega/PCA_HW_SUM_metrics.json",
    "PCA_LDA_HW": "BTA/metrics/atmega/PCA_LDA_HW_SUM_metrics.json",
    "LDA":        "BTA/metrics/atmega/LDA_SUM_metrics.json",
    "LDA_HW":     "BTA/metrics/atmega/LDA_HW_SUM_metrics.json"
}

def normalize_value(val, key):
    """
    Normalizzazione Ibrida: Logaritmica per le Traces, Lineare per il resto.
    """
    b_min, b_max = UNIVERSAL_REFERENCE_BOUNDS[key]
    val = max(b_min, min(b_max, val))
    
    if key == "Traces":
        log_val = math.log10(val)
        log_min = math.log10(b_min)
        log_max = math.log10(b_max)
        return 0.1 + 0.9 * (log_val - log_min) / (log_max - log_min)
    else:
        return 0.1 + 0.9 * (val - b_min) / (b_max - b_min)

def calculate_omega(r, w):
    total_area = 0.0
    for i in range(5):
        r_curr = r[i]
        r_next = r[(i + 1) % 5]
        sector_area = 0.5 * r_curr * r_next * sin_72
        total_area += w[i] * sector_area
    return total_area

print("="*85)
print("   ELABORAZIONE DINAMICA E UNIVERSALE PARETO OMEGA - MODELLI BTA")
print("   [!] Attiva Normalizzazione Logaritmica sull'asse delle Tracce")
print("="*85)

for model_name, file_rel_path in bta_paths.items():
    file_path = os.path.join(PROJECT_ROOT, file_rel_path)
    if not os.path.exists(file_path):
        print(f"[-] Errore: File non trovato in {file_rel_path} -> Salto modello.")
        continue
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            m_data = json.load(f)
            
        ge_curve = m_data.get("GE_curve", [])
        traces = 10000.0  
        START_OFFSET = 10
        for idx, val in enumerate(ge_curve):
            if val < 0.5:
                traces = float(idx + START_OFFSET)
                break
                
        train_time = float(m_data.get("Training_time", 0.0))
        attack_time = float(m_data.get("Attack_time", 0.0))
        ram = float(m_data.get("Training_RAM_usage", 0.0))
        
        storage_bytes = float(m_data.get("Storage_size", 0.0))
        storage_kb = storage_bytes / 1024.0
        
        r = [
            normalize_value(traces, "Traces"),
            normalize_value(train_time, "TrainTime"),
            normalize_value(attack_time, "AttackTime"),
            normalize_value(ram, "RAM"),
            normalize_value(storage_kb, "Storage")
        ]
        
        omega = calculate_omega(r, SECTOR_WEIGHTS)
        
        print(f"\nModello: {model_name}")
        print(f"  Dati Reali Estratti: Traces={traces}, TrainTime={train_time:.2f}s, AttackTime={attack_time:.2f}s, RAM={ram:.2f}MB, Storage={storage_kb:.2f}KB")
        print(f"  Vettore r_i Norm:    {[round(x, 3) for x in r]}")
        print(f"  Inefficienza Omega:  \033[1;32m{omega:.4f}\033[0m")
        
        tikz_coords = []
        for i in range(5):
            angle = 90 - i * 72
            tikz_coords.append(f"({angle}:{r[i]:.2f}*4)")
            
        print(f"  Codice TikZ:")
        print(f"    \\draw[color{model_name}, very thick, fill=color{model_name}, fill opacity=0.10] " + " -- ".join(tikz_coords) + " -- cycle;")
        
    except Exception as e:
        print(f"[-] Errore nell'elaborazione del file {file_rel_path}: {e}")

print("="*85 + "\n")
import os
import re
import json
import csv
import math
from pathlib import Path

# Angolo di un pentagono regolare (72 gradi in radianti)
sin_72 = math.sin(math.radians(72))

# ============================================================================
# CONFIGURAZIONE STRATEGICA: LIMITI DI RIFERIMENTO FISICI UNIFICATI
# ============================================================================
UNIVERSAL_REFERENCE_BOUNDS = {
    "Traces":     [10.0, 10000.0],    
    "TrainTime":  [0.0, 25000.0],     
    "AttackTime": [0.0, 30.0],        
    "RAM":        [0.0, 4500.0],      
    "Storage":    [0.0, 4500.0]       
}

SECTOR_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]

# Impostazione automatica della root del progetto
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "utils":
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)

# Percorsi dei file di log fisici per il caricamento dinamico
BTA_JSON_PATH = os.path.join(PROJECT_ROOT, "BTA", "metrics", "riscure_pinata", "PCA_LDA_SUM_metrics.json")
MLP_CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "riscure", "MLP", "nn_mlp_riscure_pinata_consolidated_metrics.csv")
CNN_CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "riscure", "CNN", "nn_cnn_zaid_riscure_pinata_consolidated_metrics.csv")
MLP_PRE_CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "riscure", "MLP_PRE", "nn_mlp_pre_riscure_pinata_consolidated_metrics.csv")

# ============================================================================
# DIZIONARIO DI FALLBACK UNIFICATO (Previene qualsiasi KeyError)
# ============================================================================
ATMEGA_SHOWDOWN_FALLBACK = {
    "BTA:PCA":       [1647.0, 0.87, 22.52, 6.05, 220.0],
    "Standard MLP":  [1093.0, 9718.31, 20.22, 1728.18, 3497.43],
    "CNN Zaid":      [80.0, 9759.72, 20.00, 1954.36, 3495.87],
    "MLP_PRE":       [13.0, 1336.8, 15.4, 3249.3, 271.9]
}

def normalize_value(val, key):
    """
    Normalizzazione Ibrida: Logaritmica per le Tracce, Lineare per le Risorse Hardware.
    Mappa i valori nell'intervallo [0.1, 1.0].
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

def get_dynamic_metrics():
    metrics = {}
    
    # 1. Caricamento dinamico BTA da JSON
    if os.path.exists(BTA_JSON_PATH):
        try:
            with open(BTA_JSON_PATH, 'r', encoding='utf-8') as f:
                bta_data = json.load(f)
            ge_curve = bta_data.get("GE_curve", [])
            traces = 10000.0
            for idx, val in enumerate(ge_curve):
                if val < 0.5:
                    traces = float(idx + 10) 
                    break
            
            train_time = float(bta_data.get("Training_time", 0.87))
            attack_time = float(bta_data.get("Attack_time", 22.52))
            ram = float(bta_data.get("Training_RAM_usage", 6.05))
            storage = float(bta_data.get("Storage_size", 225280.0)) / 1024.0 # In KB
            metrics["BTA:PCA"] = [traces, train_time, attack_time, ram, storage]
            print("[+] Caricati dinamicamente dati BTA da JSON.")
        except Exception as e:
            print(f"[-] Errore BTA JSON: {e}. Uso fallback.")
            metrics["BTA:PCA"] = ATMEGA_SHOWDOWN_FALLBACK["BTA:PCA"]
    else:
        print("[-] File BTA JSON non trovato. Uso fallback.")
        metrics["BTA:PCA"] = ATMEGA_SHOWDOWN_FALLBACK["BTA:PCA"]

    # 2. Caricamento Standard MLP da CSV (SBOX_OUT_binary_out_scalar)
    mlp_loaded = False
    if os.path.exists(MLP_CSV_PATH):
        try:
            with open(MLP_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Target"] == "SBOX_OUT" and row["Config"] == "binary_out_none":
                        traces = float(row["DataComplexity_Traces"]) if row["DataComplexity_Traces"] != "N/A" else 1093.0
                        train_time = float(row["TrainTime_s"])
                        attack_time = float(row["AttackTime_s"])
                        ram = float(row["TrainRAM_MB"])
                        storage = float(row["ModelSize_KB"])
                        metrics["Standard MLP"] = [traces, train_time, attack_time, ram, storage]
                        mlp_loaded = True
                        print("[+] Caricati dinamicamente dati Standard MLP da CSV.")
                        break
        except Exception as e:
            print(f"[-] Errore Standard MLP CSV: {e}")
    
    if not mlp_loaded:
        print("[-] File Standard MLP CSV non caricato. Uso fallback.")
        metrics["Standard MLP"] = ATMEGA_SHOWDOWN_FALLBACK["Standard MLP"]

    # 3. Caricamento CNN Zaid da CSV (SBOX_OUT_binary_out_scalar)
    cnn_loaded = False
    if os.path.exists(CNN_CSV_PATH):
        try:
            with open(CNN_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Target"] == "SBOX_OUT" and row["Config"] == "categorical_out_none":
                        traces = float(row["DataComplexity_Traces"]) if row["DataComplexity_Traces"] != "N/A" else 80.0
                        train_time = float(row["TrainTime_s"])
                        attack_time = float(row["AttackTime_s"])
                        ram = float(row["TrainRAM_MB"])
                        storage = float(row["ModelSize_KB"])
                        metrics["CNN Zaid"] = [traces, train_time, attack_time, ram, storage]
                        cnn_loaded = True
                        print("[+] Caricati dinamicamente dati CNN Zaid da CSV.")
                        break
        except Exception as e:
            print(f"[-] Errore CNN Zaid CSV: {e}")
            
    if not cnn_loaded:
        print("[-] File CNN Zaid CSV non caricato. Uso fallback.")
        metrics["CNN Zaid"] = ATMEGA_SHOWDOWN_FALLBACK["CNN Zaid"]

    # 4. Caricamento MLP_PRE da CSV (SBOX_OUT_binary_out_none)
    pre_loaded = False
    if os.path.exists(MLP_PRE_CSV_PATH):
        try:
            with open(MLP_PRE_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Target"] == "HW_SO" and row["Config"] == "categorical_out_none":
                        traces = float(row["DataComplexity_Traces"]) if row["DataComplexity_Traces"] != "N/A" else 2236.0
                        train_time = float(row["TrainTime_s"])
                        attack_time = float(row["AttackTime_s"])
                        ram = float(row["TrainRAM_MB"])
                        storage = float(row["ModelSize_KB"])
                        metrics["MLP_PRE"] = [traces, train_time, attack_time, ram, storage]
                        pre_loaded = True
                        print("[+] Caricati dinamicamente dati MLP_PRE da CSV.")
                        break
        except Exception as e:
            print(f"[-] Errore MLP_PRE CSV: {e}")
            
    if not pre_loaded:
        print("[-] File MLP_PRE CSV non caricato. Uso fallback.")
        metrics["MLP_PRE"] = ATMEGA_SHOWDOWN_FALLBACK["MLP_PRE"]

    return metrics

def main():
    print("="*85)
    print("   ELABORAZIONE PARETO OMEGA - SHOWDOWN FINALE IBRIDO (ATMEGA)")
    print("   [!] Attiva Normalizzazione Logaritmica sull'asse delle Tracce")
    print("="*85)
    
    showdown_data = get_dynamic_metrics()
    
    color_idx = 1
    for model_name, raw_values in showdown_data.items():
        r = [
            normalize_value(raw_values[0], "Traces"),
            normalize_value(raw_values[1], "TrainTime"),
            normalize_value(raw_values[2], "AttackTime"),
            normalize_value(raw_values[3], "RAM"),
            normalize_value(raw_values[4], "Storage")
        ]
        
        omega = calculate_omega(r, SECTOR_WEIGHTS)
        
        print(f"\nModello: {model_name}")
        print(f"  Dati Fisici Estratti: Traces={raw_values[0]:.1f}, TrainTime={raw_values[1]:.2f}s, AttackTime={raw_values[2]:.2f}s, RAM={raw_values[3]:.2f}MB, Storage={raw_values[4]:.2f}KB")
        print(f"  Vettore r_i Norm:     {[round(x, 3) for x in r]}")
        print(f"  Inefficienza Omega:   \033[1;32m{omega:.4f}\033[0m")
        
        # Generazione coordinate TikZ con formula (90:0.20*4) usando angoli positivi (90, 18, 306, 234, 162)
        tikz_coords_formula = []
        # Generazione coordinate TikZ con calcolo esplicito (90:0.80) per prevenire il bug del canvas bianco
        tikz_coords_precalc = []
        
        for i in range(5):
            angle = (90 - i * 72) % 360
            scaled_r = r[i] * 4.0
            tikz_coords_formula.append(f"({angle}:{r[i]:.2f}*4)")
            tikz_coords_precalc.append(f"({angle}:{scaled_r:.2f})")
            
        clean_model_name = model_name.replace(":", "").replace(" ", "_").replace("-", "_")
        
        print(f"  Codice TikZ (Espressione Matematica):")
        print(f"    \\draw[color{clean_model_name}, very thick, fill=color{clean_model_name}, fill opacity=0.10] " + " -- ".join(tikz_coords_formula) + " -- cycle;")
        print(f"  Codice TikZ (Pre-calcolato per Overleaf - CONSIGLIATO):")
        print(f"    \\draw[color{color_idx}, very thick, fill=color{color_idx}, fill opacity=0.10] " + " -- ".join(tikz_coords_precalc) + " -- cycle;")
        
        color_idx = (color_idx % 8) + 1

    print("="*85 + "\n")

if __name__ == "__main__":
    main()
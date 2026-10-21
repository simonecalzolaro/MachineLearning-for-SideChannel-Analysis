import os
import re
import csv

# ============================================================================
# CONFIGURAZIONE UTENTE: SELEZIONA PLATFORMA E ARCHITETTURA QUI
# ============================================================================
PLATFORM = "atmega"  # Opzioni: "riscure_pinata", "atmega"
ARCH = "CNN_ZAID"                # Opzioni: "MLP", "CNN"

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# Calcoliamo il percorso base dinamico della scansione
NN_BASE_DIR = os.path.join(PROJECT_ROOT, "NN", ARCH, PLATFORM)
OUTPUT_CSV = os.path.join(PROJECT_ROOT, f"nn_{ARCH.lower()}_{PLATFORM}_consolidated_metrics.csv")

def extract_numeric_value(filepath):
    """
    Legge un file txt ed estrae il primo valore numerico floating-point trovato.
    Funziona per RAM (es. 'Peak RAM overhead: 327.92 MB') e Tempi (es. 'Elapsed time: 12.99 s').
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Regex per trovare numeri interi o float (es. 12, 12.99, 1.2e+3)
            match = re.search(r"[-+]?\d*\.\d+|\d+", content)
            if match:
                return float(match.group())
    except Exception as e:
        print(f"[-] Errore nella lettura del file {filepath}: {e}")
    return None

def find_crossing_point(ge_csv_path):
    """
    Analizza il file ge.csv e restituisce il numero di traccia (1-based)
    in cui il valore di Guessing Entropy scende per la prima volta sotto 0.5.
    """
    if not os.path.exists(ge_csv_path):
        return "N/A"
    try:
        with open(ge_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            ge_values = []
            for row in reader:
                if row and row[0].strip():
                    try:
                        ge_values.append(float(row[0].strip()))
                    except ValueError:
                        continue # Salta righe di testo/intestazione
            
            for idx, val in enumerate(ge_values):
                if val < 0.5:
                    return idx + 1 # Traccia reale (1-based)
            return f">{len(ge_values)}" if ge_values else "Non converge"
    except Exception as e:
        print(f"[-] Errore nel calcolo della soglia in {ge_csv_path}: {e}")
    return "N/A"

def collect_metrics():
    attack_root = os.path.join(NN_BASE_DIR, "ATTACK")
    training_root = os.path.join(NN_BASE_DIR, "TRAINING")
    
    if not os.path.exists(attack_root) or not os.path.exists(training_root):
        print(f"[-] Errore: Assicurati che i percorsi esistano:\n - ATTACK: {attack_root}\n - TRAINING: {training_root}")
        return []
        
    records = []
    
    # Scansioniamo la cartella ATTACK per identificare i target (HW_SO, SBOX_OUT)
    for target in ["HW_SO", "SBOX_OUT"]:
        target_attack_dir = os.path.join(attack_root, target)
        if not os.path.exists(target_attack_dir):
            continue
            
        # Troviamo tutte le cartelle di configurazione (es. fixed_key_categorical_out_binary_ptx)
        for subfolder in os.listdir(target_attack_dir):
            if not subfolder.startswith("fixed_key_"):
                continue
                
            config_name = subfolder.replace("fixed_key_", "").replace("_ptx", "")
            
            # Percorsi delle directory correnti per configurazione
            attack_cfg_dir = os.path.join(target_attack_dir, subfolder)
            training_cfg_dir = os.path.join(training_root, target, subfolder)
            
            # Inizializziamo il dizionario delle metriche per questa riga
            row_metrics = {
                "Target": target,
                "Config": config_name,
                "DataComplexity_Traces": "N/A",
                "TrainTime_s": "N/A",
                "TrainRAM_MB": "N/A",
                "AttackTime_s": "N/A",
                "AttackRAM_MB": "N/A",
                "ModelSize_KB": "N/A"
            }
            
            # --- 1. Estrazione metriche di ATTACK ---
            if os.path.exists(attack_cfg_dir):
                for file in os.listdir(attack_cfg_dir):
                    file_path = os.path.join(attack_cfg_dir, file)
                    if file.endswith("ge.csv"):
                        row_metrics["DataComplexity_Traces"] = find_crossing_point(file_path)
                    elif file.endswith("attack_time.txt"):
                        row_metrics["AttackTime_s"] = extract_numeric_value(file_path)
                    elif file.endswith("attack_ram.txt"):
                        row_metrics["AttackRAM_MB"] = extract_numeric_value(file_path)
            
            # --- 2. Estrazione metriche di TRAINING ---
            if os.path.exists(training_cfg_dir):
                for file in os.listdir(training_cfg_dir):
                    file_path = os.path.join(training_cfg_dir, file)
                    if file.endswith("training_time.txt"):
                        row_metrics["TrainTime_s"] = extract_numeric_value(file_path)
                    elif file.endswith("training_ram.txt"):
                        row_metrics["TrainRAM_MB"] = extract_numeric_value(file_path)
                    elif file.endswith("model.h5"):
                        # Calcolo dinamico dello Storage in Kilobyte
                        bytes_size = os.path.getsize(file_path)
                        row_metrics["ModelSize_KB"] = round(bytes_size / 1024.0, 2)
            
            records.append(row_metrics)
            print(f"[+] Estratte metriche per: {target:<10} | {config_name:<30}")
            
    return records

def main():
    print("="*80)
    print(f"Scansione in corso per la tesi: {ARCH} su {PLATFORM}")
    print(f"Cartella di origine: {NN_BASE_DIR}")
    print("="*80)
    
    data = collect_metrics()
    
    if not data:
        print("[-] Nessun dato estratto. Verifica i parametri PLATFORM e ARCH.")
        return
        
    # Campi di intestazione del CSV
    headers = [
        "Target", "Config", "DataComplexity_Traces", 
        "TrainTime_s", "TrainRAM_MB", 
        "AttackTime_s", "AttackRAM_MB", "ModelSize_KB"
    ]
    
    # Scrittura su file CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        
    print("="*80)
    print(f"COMPLETATO! File di riepilogo generato con successo:")
    print(f" -> {OUTPUT_CSV}")
    print("="*80)

if __name__ == "__main__":
    main()
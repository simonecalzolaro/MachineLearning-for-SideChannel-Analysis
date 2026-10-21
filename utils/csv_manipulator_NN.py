import os
import csv

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# Lista dei percorsi forniti (relativi alla radice del progetto)
paths = [
    "NN/CNN_ZAID/atmega/ATTACK/HW_SO/fixed_key_categorical_out_binary_ptx/HW_SO_ptx_binary_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/HW_SO/fixed_key_categorical_out_none_ptx/HW_SO_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/HW_SO/fixed_key_categorical_out_scalar_ptx/HW_SO_ptx_scalar_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/SBOX_OUT/fixed_key_binary_out_binary_ptx/SBOX_OUT_ptx_binary_out_binary_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/SBOX_OUT/fixed_key_binary_out_none_ptx/SBOX_OUT_out_binary_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/SBOX_OUT/fixed_key_binary_out_scalar_ptx/SBOX_OUT_ptx_scalar_out_binary_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/SBOX_OUT/fixed_key_categorical_out_binary_ptx/SBOX_OUT_ptx_binary_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/SBOX_OUT/fixed_key_categorical_out_none_ptx/SBOX_OUT_ge.csv",
    "NN/CNN_ZAID/atmega/ATTACK/SBOX_OUT/fixed_key_categorical_out_scalar_ptx/SBOX_OUT_ptx_scalar_ge.csv"
]

# Impostiamo il fattore di sottocampionamento (prendiamo 1 punto ogni 10)
DOWNSAMPLE_STEP = 10
OUTPUT_DIR = os.path.join("csv", "downsampled")

print("\n" + "="*80)
print("   ELABORAZIONE FILE DI ABLAZIONE NN (DOWN SAMPLING & RILEVAMENTO SOGLIA GE < 0.5)")
print("="*80)

for file_rel_path in paths:
    file_path = os.path.join(PROJECT_ROOT, file_rel_path)
    
    if not os.path.exists(file_path):
        print(f"[-] File non trovato: {file_rel_path} -> Salto.")
        continue
    
    # Generiamo un nome descrittivo e flat basato sulle cartelle del percorso per evitare collisioni
    parts = file_rel_path.split('/')
    target = parts[4]  # es. HW_SO o SBOX_OUT
    config = parts[5].replace("fixed_key_", "").replace("_ptx", "") # es. categorical_out_binary
    
    out_filename = f"{target}_{config}_downsampled.csv"
    out_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, out_filename)
    
    # Leggiamo i dati dal CSV originale
    ge_values = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].strip():
                try:
                    # Converte la notazione scientifica (es. 2.24e+02) in float
                    ge_values.append(float(row[0].strip()))
                except ValueError:
                    continue # Salta righe di testo non numeriche se presenti
    
    if not ge_values:
        print(f"[-] Nessun valore valido trovato in: {file_rel_path}")
        continue
        
    # Calcoliamo il punto esatto di incrocio (GE < 0.5) - 1-based index (numero di traccia reale)
    crossing_point = "Non converge (>1000)"
    for idx, val in enumerate(ge_values):
        if val < 0.5:
            crossing_point = f"Traccia {idx + 1}"
            break
            
    print(f"[+] Modello: {target:<10} | Config: {config:<30} | Successo: {crossing_point}")
    
    # Generiamo i dati sotto-campionati mantenendo la traccia originale come asse X
    downsampled_rows = []
    for idx in range(0, len(ge_values), DOWNSAMPLE_STEP):
        downsampled_rows.append([idx + 1, ge_values[idx]])
        
    # Assicuriamoci che l'ultimissimo punto della simulazione sia incluso nel grafico
    last_idx = len(ge_values) - 1
    if last_idx % DOWNSAMPLE_STEP != 0:
        downsampled_rows.append([last_idx + 1, ge_values[last_idx]])
        
    # Scriviamo il nuovo file CSV strutturato a due colonne (Traces, GE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Traces", "GE"])  # Intestazioni esplicite per LaTeX
        writer.writerows(downsampled_rows)

print("="*80)
print(f"Successo! Tutti i file sotto-campionati sono stati salvati in: '{OUTPUT_DIR}'")
print("="*80 + "\n")
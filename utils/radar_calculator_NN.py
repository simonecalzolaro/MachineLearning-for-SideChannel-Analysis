import os
import math
import csv

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# Definizione della costante trigonometrica per l'angolo tra gli assi di un pentagono (72 gradi)
sin_72 = math.sin(math.radians(72))

# Percorso relativo per il file CSV generato precedentemente
CSV_PATH = os.path.join("csv", "atmega", "MLP_PRE", "nn_mlp_pre_atmega_consolidated_metrics.csv")

# 1. DATI REALI DI FALLBACK (Sincronizzati con i 5 assi: [Traces, TrainTime (s), AttackTime (s), RAM (MB), Storage (KB)])
fallback_data = {
    "BTA:PCA_LDA":   [12, 17.877785384212622, 11.33925576438196, 2217.11328125, 24.576], # Ripristinato il valore 12 (Traces)
    "Standard MLP": [268, 19121.6, 14.42, 2774.25, 3156.8],                           # Corretto ordine RAM/AttackTime
    "CNN":          [32, 22062.13, 14.08, 2779.7, 322.72],                             # Corretto ordine RAM/AttackTime
    "Custom MLP":   [10, 2645.0, 20.9, 3578.9, 378.0]                                  # Coerente con la configurazione
}

def load_metrics_from_csv(csv_path):
    """
    Tenta di caricare i dati dal CSV consolidato. Se non disponibile,
    esegue il fallback sul dizionario statico preimpostato.
    """
    if not os.path.exists(csv_path):
        print(f"[-] CSV non trovato in '{csv_path}'. Utilizzo dei dati reali di fallback...")
        return fallback_data
    
    data = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = row["Target"]
                config = row["Config"]
                key = f"{target}_{config}"
                
                # Converte i valori gestendo eventuali anomalie 'N/A' con massimali di sicurezza
                try:
                    traces = float(row["DataComplexity_Traces"]) if row["DataComplexity_Traces"] != "N/A" else 1000.0
                    train_time = float(row["TrainTime_s"]) if row["TrainTime_s"] != "N/A" else 20000.0
                    attack_time = float(row["AttackTime_s"]) if row["AttackTime_s"] != "N/A" else 20.0
                    train_ram = float(row["TrainRAM_MB"]) if row["TrainRAM_MB"] != "N/A" else 2000.0
                    storage = float(row["ModelSize_KB"]) if row["ModelSize_KB"] != "N/A" else 4000.0
                    
                    data[key] = [traces, train_time, attack_time, train_ram, storage]
                except ValueError:
                    continue
    except Exception as e:
        print(f"[-] Errore nella lettura del CSV ({e}). Avvio fallback dinamico...")
        return fallback_data
    return data

def normalize_metrics(raw_data):
    """
    Normalizza linearmente ciascuna metrica in modo indipendente nell'intervallo [0.1, 1.0].
    """
    metrics_keys = ["Traces", "TrainTime", "AttackTime", "RAM", "Storage"]
    normalized_by_metric = {}
    
    # Estraiamo le colonne per normalizzarle ciascuna sul proprio asse
    for col_idx, metric_name in enumerate(metrics_keys):
        col_values = [raw_data[model][col_idx] for model in raw_data]
        min_v = min(col_values)
        max_v = max(col_values)
        
        if max_v == min_v:
            normalized_by_metric[metric_name] = [0.1 for _ in col_values]
        else:
            normalized_by_metric[metric_name] = [
                0.1 + 0.9 * (v - min_v) / (max_v - min_v) for v in col_values
            ]

    # Ricostruiamo la struttura per ciascun modello
    normalized_models = {}
    for row_idx, model in enumerate(raw_data.keys()):
        normalized_models[model] = [
            normalized_by_metric["Traces"][row_idx],
            normalized_by_metric["TrainTime"][row_idx],
            normalized_by_metric["AttackTime"][row_idx],
            normalized_by_metric["RAM"][row_idx],
            normalized_by_metric["Storage"][row_idx]
        ]
            
    return normalized_models

def calculate_omega(r, w):
    """
    Calcola l'area pesata del poligono di inefficienza scomponendolo in 5 triangoli.
    """
    total_area = 0.0
    for i in range(5):
        r_curr = r[i]
        r_next = r[(i + 1) % 5]
        
        # Area geometrica del singolo spicchio polare
        sector_area = 0.5 * r_curr * r_next * sin_72
        # Moltiplicazione per il rispettivo peso asimmetrico
        total_area += w[i] * sector_area
    return total_area

def main():
    print("="*80)
    print("   CALCOLO INEFFICIENZA GLOBALE OMEGA (PENTAGONO PARETO RADAR)")
    print("="*80)
    
    # Caricamento ed elaborazione dati
    raw_data = load_metrics_from_csv(CSV_PATH)
    normalized_data = normalize_metrics(raw_data)
    
    # Distribuzione asimmetrica dei pesi: 60% sul canale primario (Traces), 10% sui restanti 4
    weights = [0.6, 0.1, 0.1, 0.1, 0.1]
    
    # Modelli di interesse per la tesi (sia in modalità CSV che in modalità Fallback)
    allowed_models = [
        "SBOX_OUT_categorical_out_none", "SBOX_OUT_binary_out_none",
        "SBOX_OUT_binary_out_scalar", "SBOX_OUT_binary_out_binary",
        "SBOX_OUT_categorical_out_scalar", "SBOX_OUT_categorical_out_binary"

        
    ]
    
    color_index = 1
    for model, r in normalized_data.items():
        if model not in allowed_models:
            continue
            
        omega = calculate_omega(r, weights)
        print(f"\nModello: {model}")
        print(f"  Vettore r_i normalizzato: {[round(x, 3) for x in r]}")
        print(f"  Area Pesata Inefficienza \\Omega: {omega:.4f}")
        
        # Generazione coordinate TikZ centrate su scala 4.0
        tikz_coords = []
        for i in range(5):
            angle = 90 - i * 72
            tikz_coords.append(f"({angle}:{r[i]:.2f}*4)")
        
        # Generiamo due opzioni di stampa per facilitare l'importazione nel documento LaTeX
        clean_model_name = model.replace(":", "").replace(" ", "_").replace("+", "_")
        print(f"  Codice TikZ (Colori Semantici):\n  \\draw[color{clean_model_name}, very thick, fill=color{clean_model_name}, fill opacity=0.10] " + " -- ".join(tikz_coords) + " -- cycle;")
        print(f"  Codice TikZ (Colori Indicizzati per Overleaf):\n  \\draw[color{color_index}, very thick, fill=color{color_index}, fill opacity=0.10] " + " -- ".join(tikz_coords) + " -- cycle;")
        
        color_index = (color_index % 6) + 1  # Cicla tra i 6 colori principali del preambolo LaTeX
        
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
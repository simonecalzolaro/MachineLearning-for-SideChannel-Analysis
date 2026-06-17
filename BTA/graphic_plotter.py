import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

# Risolve l'errore ModuleNotFoundError trovando dinamicamente la cartella utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
os.chdir(PROJECT_ROOT)

# Importiamo le costanti. Ho rimosso WEIGHT_GE perché useremo i pesi unificati
from constants import BTA_METRICS_PATH_ATMEGA, BTA_METRICS_PATH_RISCURE_PINATA, PATH_PLOTS_ATMEGA, PATH_PLOTS_RISCURE_PINATA, MAX_TRACES

# ==========================================
# CONFIGURAZIONI GLOBALI E PESI (Allineati con NN)
# ==========================================
features_choice = ['PCA_', 'LDA_', 'PCA_LDA_', 'PCA_HW_', 'LDA_HW_', 'PCA_LDA_HW_']
operations = ['SUM']
end_base = "_metrics.json"

WEIGHTS = {
    "traces": 0.30,
    "attack_time": 0.175,
    "train_time": 0.175,
    "ram": 0.175,
    "storage": 0.175
}

def get_color(f_c):
    base_colors = {'PCA': '#1f77b4', 'LDA': '#ff7f0e', 'PCA LDA': '#2ca02c', 
                   'PCA HW': '#d62728', 'LDA HW': '#9467bd', 'PCA LDA HW': '#8c564b'}
    return base_colors.get(f_c, 'gray')

# ==========================================
# 1. TEMPI COMPUTAZIONALI
# ==========================================
def plot_all_times(experiment_data, features_choice, results_path):
    models, train_times, attack_times = [], [], []

    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            clean_name = f_c.replace('_', ' ').strip().upper()
            models.append(clean_name)
            train_times.append(experiment_data[dict_key]['Training_time'])
            attack_times.append(experiment_data[dict_key]['Attack_time'])

    if not models: return

    x_positions = np.arange(len(models))
    bar_width = 0.35
    plt.figure(figsize=(12, 6)) 
    
    bars_train = plt.bar(x_positions - bar_width/2, train_times, bar_width, label='Training Time', color='#1f77b4', edgecolor='black')
    bars_attack = plt.bar(x_positions + bar_width/2, attack_times, bar_width, label='Attack Time', color='#ff7f0e', edgecolor='black')

    for bars in [bars_train, bars_attack]:
        for bar in bars:
            yval = bar.get_height()
            if yval > 0.01:
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max(train_times + attack_times) * 0.01), 
                         f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.xlabel('Feature Reduction Models', fontsize=12, fontweight='bold')
    plt.ylabel('Time (Seconds)', fontsize=12, fontweight='bold')
    plt.title('Computational Time Comparison', fontsize=14)
    plt.xticks(x_positions, models, fontsize=10, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    out_dir = os.path.join(results_path, "TIME")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Time_Comparison_All.png"), dpi=300)
    plt.close()

# ==========================================
# 2. GRAFICI RAGGRUPPATI (SBOX vs HW)
# ==========================================
def plot_grouped_metric(experiment_data, metric_key, title, ylabel, results_path, is_time=False):
    base_models = ['PCA', 'LDA', 'PCA_LDA']
    x_labels = ['PCA', 'LDA', 'PCA + LDA']
    
    variations = [
        (False, 'SBOX Target', '#1f77b4'),
        (True,  'HW Target', '#d62728')
    ]

    x_positions = np.arange(len(base_models))
    bar_width = 0.35
    plt.figure(figsize=(10, 6))
    all_vals = []

    for i, (is_hw, label, color) in enumerate(variations):
        offset = (i - 0.5) * bar_width
        y_values = []
        
        for base in base_models:
            f_c = f"{base}_HW_" if is_hw else f"{base}_"
            dict_key = f"{f_c}SUM"
            
            if dict_key in experiment_data:
                val = experiment_data[dict_key][metric_key]
                y_values.append(val)
                all_vals.append(val)
            else:
                y_values.append(0)

        bars = plt.bar(x_positions + offset, y_values, bar_width, label=label, color=color, edgecolor='black')

        for bar in bars:
            yval = bar.get_height()
            if yval > 0 and all_vals:
                fmt = f"{yval:.2f}s" if is_time else f"{yval:.1f}"
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max(all_vals) * 0.02), 
                         fmt, ha='center', va='bottom', fontweight='bold', color='black', fontsize=10)

    plt.xlabel('Feature Reduction Architecture', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=15)
    plt.xticks(x_positions, x_labels, fontsize=11, fontweight='bold')
    plt.legend(title="Leakage Model", loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    folder = "TIME" if is_time else "RAM"
    out_dir = os.path.join(results_path, folder)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"Bar_Grouped_{metric_key}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. RAM E STORAGE
# ==========================================
def plot_all_ram(experiment_data, features_choice, results_path):
    models, train_ram, attack_ram = [], [], []
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            models.append(f_c.replace('_', ' ').strip().upper())
            train_ram.append(experiment_data[dict_key]['Training_RAM_usage'])
            attack_ram.append(experiment_data[dict_key]['Attack_RAM_usage'])

    if not models: return
    x_positions = np.arange(len(models))
    bar_width = 0.35
    plt.figure(figsize=(12, 6))
    
    bars_train = plt.bar(x_positions - bar_width/2, train_ram, bar_width, label='Training RAM', color='#1f77b4', edgecolor='black')
    bars_attack = plt.bar(x_positions + bar_width/2, attack_ram, bar_width, label='Attack RAM', color='#ff7f0e', edgecolor='black')

    for bars in [bars_train, bars_attack]:
        for bar in bars:
            yval = bar.get_height()
            if yval > 0.1:
                plt.text(bar.get_x() + bar.get_width()/2, yval + (max(train_ram + attack_ram) * 0.01), 
                         f"{yval:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=8)

    plt.xlabel('Feature Reduction Models', fontsize=12, fontweight='bold')
    plt.ylabel('Peak Memory Usage (MB)', fontsize=12, fontweight='bold')
    plt.title('RAM Footprint Comparison', fontsize=14)
    plt.xticks(x_positions, models, fontsize=10, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    out_dir = os.path.join(results_path, "RAM")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "RAM_Comparison_All.png"), dpi=300)
    plt.close()

def plot_all_storage(experiment_data, features_choice, results_path):
    models, storage_kb, colors = [], [], []
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            clean_name = f_c.replace('_', ' ').strip().upper()
            models.append(clean_name)
            storage_kb.append(experiment_data[dict_key]['Storage_size'] / 1024.0)
            colors.append(get_color(clean_name))

    if not models: return
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, storage_kb, color=colors, edgecolor='black', linewidth=1.2)

    for bar in bars:
        yval = bar.get_height()
        if yval > 0.1:
            plt.text(bar.get_x() + bar.get_width()/2, yval + (max(storage_kb) * 0.01), 
                     f"{yval:.1f} KB", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.title('Template Storage Size Comparison', fontsize=14)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    out_dir = os.path.join(results_path, "STORAGE")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Storage_Comparison_All.png"), dpi=300)
    plt.close()

# ==========================================
# 4. TRACCE E GE CURVES
# ==========================================
def plot_all_traces(experiment_data, features_choice, results_path):
    models, traces, colors = [], [], []
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            clean_name = f_c.replace('_', ' ').strip().upper()
            models.append(clean_name)
            val = min(experiment_data[dict_key]['GE_crossing_point'], MAX_TRACES)
            traces.append(val)
            colors.append(get_color(clean_name))

    if not models: return
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, traces, color=colors, edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, traces):
        yval = bar.get_height()
        text = f">{MAX_TRACES}" if val >= MAX_TRACES else str(int(val))
        text_color = 'red' if val >= MAX_TRACES else 'black'
        plt.text(bar.get_x() + bar.get_width()/2, yval + (MAX_TRACES * 0.02), 
                 text, ha='center', va='bottom', fontweight='bold', fontsize=11, color=text_color)

    plt.axhline(y=MAX_TRACES, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Max Traces Evaluated ({MAX_TRACES})')
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Minimum Traces (GE < 0.5)', fontsize=12, fontweight='bold')
    plt.title('Attack Efficiency: Traces Required', fontsize=14)
    plt.ylim(0, MAX_TRACES * 1.1)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    out_dir = os.path.join(results_path, "TRACES")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Traces_Comparison_All.png"), dpi=300)
    plt.close()

def plot_all_ge_curves(experiment_data, features_choice, results_path):
    plt.figure(figsize=(12, 7))
    plotted_any = False
    
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            cp = experiment_data[dict_key]['GE_crossing_point']
            if cp >= MAX_TRACES: continue 
            ge_curve = experiment_data[dict_key]['GE_curve'][:MAX_TRACES]
            clean_name = f_c.replace('_', ' ').strip().upper()
            
            plt.plot(ge_curve, label=f"{clean_name} | GE < 0.5: {int(cp)}", 
                     color=get_color(clean_name), linewidth=2.5)
            plotted_any = True

    if not plotted_any: return
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Success Threshold')
    plt.xlabel('Number of Traces', fontsize=12, fontweight='bold')
    plt.ylabel('Guessing Entropy', fontsize=12, fontweight='bold')
    plt.title('Guessing Entropy Model Comparison (Successful Attacks)', fontsize=15)
    plt.xlim(0, MAX_TRACES)
    plt.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    out_dir = os.path.join(results_path, "GE", "ge_curves")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "GE_Overlaid_Successful.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 5. MATEMATICA E RADAR CHART (Identica alle NN)
# ==========================================
def normalize_cost(values):
    vals = np.array(values, dtype=float)
    if len(vals) == 0: return vals
    min_v, max_v = np.min(vals), np.max(vals)
    if max_v == min_v: return np.ones_like(vals) if max_v > 0 else np.zeros_like(vals)
    return (vals - min_v) / (max_v - min_v)

def calculate_polygon_area(values):
    N = len(values)
    angle = 2 * pi / N
    area = sum(0.5 * values[i] * values[(i + 1) % N] * np.sin(angle) for i in range(N))
    max_area = (N / 2) * np.sin(angle)
    return area / max_area

def plot_all_radar_chart(experiment_data, features_choice, results_path):
    models_data = []
    
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            cp = min(experiment_data[dict_key]['GE_crossing_point'], MAX_TRACES)
            if cp >= MAX_TRACES:
                continue

            clean_name = f_c.replace('_', ' ').strip().upper()
            
            # Estrazione metriche (Stesso formato delle NN)
            # Nota: converto storage in MB per matchare le NN
            models_data.append({
                'id': clean_name,
                'traces': cp,
                'attack_time': experiment_data[dict_key]['Attack_time'],
                'train_time': experiment_data[dict_key]['Training_time'],
                'attack_ram': experiment_data[dict_key]['Attack_RAM_usage'],
                'train_ram': experiment_data[dict_key]['Training_RAM_usage'],
                'storage': experiment_data[dict_key]['Storage_size'] / (1024.0 * 1024.0) 
            })

    if not models_data: return

    # Calcolo punteggi e normalizzazione (Esattamente come NN)
    traces = [m['traces'] for m in models_data]
    a_times = [m['attack_time'] for m in models_data]
    t_times = [m['train_time'] for m in models_data]
    max_rams = [max(m['attack_ram'], m['train_ram']) for m in models_data]
    storages = [m['storage'] for m in models_data]

    c_traces = normalize_cost(traces)
    c_atime = normalize_cost(a_times)
    c_ttime = normalize_cost(t_times)
    c_ram = normalize_cost(max_rams)
    c_storage = normalize_cost(storages)

    for i, m in enumerate(models_data):
        m['radar_data'] = [c_traces[i], c_atime[i], c_ttime[i], c_ram[i], c_storage[i]]
        m['area'] = calculate_polygon_area(m['radar_data'])
        
    models_data = sorted(models_data, key=lambda x: x['area'])

    # Disegno grafico (Esattamente come NN)
    categories = ['Traces\n(GE<0.5)', 'Attack Time', 'Train Time', 'Peak RAM', 'Storage']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", "Pessimo\n(Costo Max)"], color="red", size=10, alpha=0.7)
    ax.text(0, 0, 'Ottimo\n(Costo Min)', horizontalalignment='center', verticalalignment='center', size=10, color='green', fontweight='bold')
    plt.ylim(0, 1.05)

    for i, m in enumerate(models_data):
        values = m['radar_data'][:]
        values.append(values[0])
        label_text = f"{m['id']} (Area: {m['area']:.3f})"
        c = get_color(m['id'])
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=label_text, color=c)
        ax.fill(angles, values, color=c, alpha=0.15)

    plt.title(f"5D Pareto Analysis: BTA Models\n(L'area MINORE indica prestazioni MIGLIORI)", size=15, y=1.12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    
    out_dir = os.path.join(results_path, "RADAR")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Radar_Pareto_Successful_Only.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================
def generate_graphs(metrics_path, results_path):
    print(f"[*] Lettura metriche da: {metrics_path}")
    print(f"[*] Salvataggio grafici in: {results_path}")
    print(f"[*] Limite Traces: {MAX_TRACES}\n")

    experiment_data = {}
    
    for f_c in features_choice:
        for op in operations:
            filename = f"{f_c}{op}{end_base}"
            filepath = os.path.join(metrics_path, filename)
            try:
                with open(filepath, 'r') as f:
                    metric = json.load(f)
                    dict_key = f"{f_c}{op}"
                    experiment_data[dict_key] = metric
            except FileNotFoundError:
                print(f"Warning: Metrics file non trovato: {filename}. Skipping.")

    if not experiment_data:
        print("[!] Nessun dato caricato. Uscita.")
        return

    plot_grouped_metric(experiment_data, 'Training_time', 'Training Time: Leakage Model Comparison', 'Time (Seconds)', results_path, is_time=True)
    plot_grouped_metric(experiment_data, 'Attack_time', 'Attack Time: Leakage Model Comparison', 'Time (Seconds)', results_path, is_time=True)
    plot_grouped_metric(experiment_data, 'Training_RAM_usage', 'Training RAM: Leakage Model Comparison', 'Peak Memory Usage (MB)', results_path)
    plot_grouped_metric(experiment_data, 'Attack_RAM_usage', 'Attack RAM: Leakage Model Comparison', 'Peak Memory Usage (MB)', results_path)

    plot_all_traces(experiment_data, features_choice, results_path)
    plot_all_times(experiment_data, features_choice, results_path)
    plot_all_ram(experiment_data, features_choice, results_path)
    plot_all_storage(experiment_data, features_choice, results_path)
    
    plot_all_ge_curves(experiment_data, features_choice, results_path)
    plot_all_radar_chart(experiment_data, features_choice, results_path)
        
    print("\n[*] Tutti i grafici BTA generati con successo!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 BTA/graphic_plotter.py [atmega|riscure_pinata]")
        sys.exit(1)

    dev = sys.argv[1]

    if dev == "atmega":
        m_path = BTA_METRICS_PATH_ATMEGA
        r_path = os.path.join(PATH_PLOTS_ATMEGA, "BTA")
        generate_graphs(m_path, r_path)
        
    elif dev == "riscure_pinata":
        m_path = BTA_METRICS_PATH_RISCURE_PINATA
        r_path = os.path.join(PATH_PLOTS_RISCURE_PINATA, "BTA")
        generate_graphs(m_path, r_path)
        
    else:
        print("[!] Target sconosciuto. Usa 'atmega' o 'riscure_pinata'.")
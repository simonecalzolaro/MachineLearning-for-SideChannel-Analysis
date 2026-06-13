import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from math import pi, sin

sys.path.insert(0, '../utils')
from constants import BTA_METRICS_PATH_ATMEGA, BTA_METRICS_PATH_RISCURE_PINATA, PATH_PLOTS_ATMEGA, PATH_PLOTS_RISCURE_PINATA, MAX_TRACES, WEIGHT_GE

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
features_choice = ['PCA_', 'LDA_', 'PCA_LDA_', 'PCA_HW_', 'LDA_HW_', 'PCA_LDA_HW_']
operations = ['SUM']
end_base = "_metrics.json"

def get_color(f_c):
    base_colors = {'PCA': '#1f77b4', 'LDA': '#ff7f0e', 'PCA LDA': '#2ca02c', 
                   'PCA HW': '#d62728', 'LDA HW': '#9467bd', 'PCA LDA HW': '#8c564b'}
    return base_colors.get(f_c, 'gray')

# ==========================================
# 1. TEMPI COMPUTAZIONALI (ALL-IN-ONE)
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
# 3. RAM E STORAGE (ALL-IN-ONE)
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
# 4. GRAFICI TRACCE (GE < 0.5) E SCATTER TRADEOFF
# ==========================================
def plot_all_traces(experiment_data, features_choice, results_path):
    models, traces, colors = [], [], []
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            clean_name = f_c.replace('_', ' ').strip().upper()
            models.append(clean_name)
            val = min(experiment_data[dict_key]['GE_crossing_point'], MAX_TRACES) # Cap al MAX_TRACES
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

def plot_all_tradeoff(experiment_data, features_choice, results_path):
    plt.figure(figsize=(10, 7))
    plotted_any = False
    
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            x_val = min(experiment_data[dict_key]['GE_crossing_point'], MAX_TRACES) # Cap al MAX_TRACES
            y_val = experiment_data[dict_key]['Attack_time']
            clean_name = f_c.replace('_', ' ').strip().upper()
            color = get_color(clean_name)

            if x_val >= MAX_TRACES:
                plt.scatter(MAX_TRACES, y_val, color=color, marker='X', s=200, edgecolor='black', zorder=5)
                plt.annotate(f"{clean_name}\n(Failed)", (MAX_TRACES, y_val), textcoords="offset points", 
                             xytext=(-15, 5), ha='right', fontsize=9, color='gray')
            else:
                plt.scatter(x_val, y_val, color=color, marker='o', s=150, edgecolor='black', zorder=5)
                plt.annotate(clean_name, (x_val, y_val), textcoords="offset points", 
                             xytext=(10, 5), ha='left', fontsize=10, fontweight='bold', color=color)
            plotted_any = True

    if not plotted_any: return
    
    plt.axvline(x=MAX_TRACES, color='red', linestyle='--', alpha=0.5, label=f'Trace Limit ({MAX_TRACES})')
    
    # Area verde pesata: se WEIGHT_GE è alto, l'area ottimale accetta meno tracce ma è più tollerante col tempo
    opt_traces = MAX_TRACES * (1.0 - WEIGHT_GE) 
    plt.axvspan(0, opt_traces, ymin=0, ymax=0.4, color='green', alpha=0.05, label=f'Optimal Zone (Weighted {WEIGHT_GE*100:.0f}% GE)')
    
    plt.xlabel('Data Complexity: Traces to GE < 0.5', fontsize=12, fontweight='bold')
    plt.ylabel('Time Complexity: Attack Time (Seconds)', fontsize=12, fontweight='bold')
    plt.title('Attack Efficiency Trade-Off (Weighted)', fontsize=15)
    plt.xlim(0, MAX_TRACES * 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    out_dir = os.path.join(results_path, "TRADE_OFF")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Scatter_TradeOff_All.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 5. GE CURVES (FILTRATO GE < MAX_TRACES)
# ==========================================
def plot_all_ge_curves(experiment_data, features_choice, results_path):
    plt.figure(figsize=(12, 7))
    plotted_any = False
    
    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            cp = experiment_data[dict_key]['GE_crossing_point']
            
            if cp >= MAX_TRACES:
                continue 

            ge_curve = experiment_data[dict_key]['GE_curve'][:MAX_TRACES] # Taglia la curva a MAX_TRACES
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
# 6. RADAR CHART PARETO (PESATO CON FLOOR)
# ==========================================
def plot_all_radar_chart(experiment_data, features_choice, results_path):
    categories = ['Traces (GE<0.5)', 'Attack Time', 'Training Time', 'Peak Attack RAM', 'Storage Size']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 

    # Calcolo pesi dinamici
    rem_weight = (1.0 - WEIGHT_GE) / (N - 1)
    weights = [WEIGHT_GE, rem_weight, rem_weight, rem_weight, rem_weight]

    raw_data = {}
    color_mapping = {}

    for f_c in features_choice:
        dict_key = f"{f_c}SUM"
        if dict_key in experiment_data:
            cp = min(experiment_data[dict_key]['GE_crossing_point'], MAX_TRACES)
            
            if cp >= MAX_TRACES:
                continue

            clean_name = f_c.replace('_', ' ').strip().upper()
            raw_data[clean_name] = [
                cp,
                experiment_data[dict_key]['Attack_time'],
                experiment_data[dict_key]['Training_time'],
                experiment_data[dict_key]['Attack_RAM_usage'],
                experiment_data[dict_key]['Storage_size']
            ]
            color_mapping[clean_name] = get_color(clean_name)

    if not raw_data: return

    all_values = np.array(list(raw_data.values()))
    mins = all_values.min(axis=0)
    maxs = all_values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1e-10 

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    weighted_categories = [f"{cat}\n({w*100:.0f}%)" for cat, w in zip(categories, weights)]
    plt.xticks(angles[:-1], weighted_categories, fontsize=11, fontweight='bold')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "Max"], color="grey", size=9)
    plt.ylim(0, 1.1) 

    plot_elements = []
    
    for model_name, values in raw_data.items():
        norm_vals = (values - mins) / ranges
        
        # AGGIUNTA FLOOR (0.1) per evitare che i modelli perfetti (0) scompaiano
        # Comprime i valori nell'intervallo [0.1, 1.0] anziché [0.0, 1.0]
        norm_vals = 0.1 + (norm_vals * 0.9)

        # APPLICAZIONE DEI PESI
        weighted_vals = norm_vals * np.array(weights) * N 
        weighted_vals = np.append(weighted_vals, weighted_vals[0])
        
        area = 0.5 * sin(2 * pi / N) * sum(weighted_vals[j] * weighted_vals[j+1] for j in range(N))
        
        c = color_mapping[model_name]
        
        line, = ax.plot(angles, weighted_vals, linewidth=2.5, linestyle='-', color=c)
        ax.fill(angles, weighted_vals, color=c, alpha=0.1)
        
        label_text = f"{model_name} (W. Area: {area:.2f})"
        plot_elements.append((area, line, label_text))

    plot_elements.sort(key=lambda x: x[0])
    sorted_lines = [item[1] for item in plot_elements]
    sorted_labels = [item[2] for item in plot_elements]

    plt.title(f'5D Pareto Efficiency (Weighted {WEIGHT_GE*100:.0f}% Traces)\n(Smaller Area = Better Overall Efficiency)', size=16, y=1.1, fontweight='bold')
    plt.legend(sorted_lines, sorted_labels, loc='upper right', bbox_to_anchor=(1.35, 1.1), title="Ranked by Efficiency")

    out_dir = os.path.join(results_path, "RADAR")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Radar_Pareto_Successful_Only.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================
def generate_graphs(metrics_path, results_path):
    print(f"[*] Lettura metriche da: {metrics_path}")
    print(f"[*] Salvataggio grafici in: {results_path}")
    print(f"[*] Limite Tracce (MAX_TRACES): {MAX_TRACES}")
    print(f"[*] Peso Guessing Entropy (WEIGHT_GE): {WEIGHT_GE*100:.0f}%\n")

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
    plot_all_tradeoff(experiment_data, features_choice, results_path)
    
    plot_all_ge_curves(experiment_data, features_choice, results_path)
    plot_all_radar_chart(experiment_data, features_choice, results_path)
        
    print("\n[*] Tutti i grafici generati con successo!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 plot_generator.py [atmega|riscure_pinata]")
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
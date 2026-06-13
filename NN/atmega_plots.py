import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

sys.path.insert(0, "./utils")
try:
    from constants import PATH_PLOTS_ATMEGA
except ImportError:
    PATH_PLOTS_ATMEGA = "plots/atmega/"

# ==========================================
# 2. CONFIGURAZIONE E PESI
# ==========================================
DEVICE = "atmega"
ARCHS = ["CNN_ZAID", "MLP", "MLP_PRE"]
TARGETS = ["HW_SO", "SBOX_OUT"]
OUT_MODES = ["categorical", "binary"] 
PTX_MODES = ["none", "scalar", "binary"]
MAX_TRACES = 500

# PESI PER IL CALCOLO DEL MIGLIOR MODELLO (Pareto)
WEIGHTS = {
    "traces": 0.50,
    "attack_time": 0.15,
    "train_time": 0.10,
    "ram": 0.15,
    "storage": 0.10
}

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 15,
    'legend.fontsize': 11, 'figure.figsize': (16, 7), # Allargato per ospitare 9 barre comode
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'
})

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#66c2a5']

# ==========================================
# 3. HELPER FUNCTIONS (RICERCA DINAMICA FILE)
# ==========================================
def get_out_dir(arch, metric=""):
    d = os.path.join(PATH_PLOTS_ATMEGA, arch, metric) if metric else os.path.join(PATH_PLOTS_ATMEGA, arch)
    os.makedirs(d, exist_ok=True)
    return d

def get_all_paths(arch, target, out_mode, ptx_mode):
    """
    Entra nella cartella e cerca i file in base all'estensione/suffisso,
    ignorando il prefisso esatto per evitare file non trovati.
    """
    folder = f"fixed_key_{out_mode}_out_{ptx_mode}_ptx"
    attack_dir = os.path.join("NN", arch, DEVICE, "ATTACK", target, folder)
    train_dir = os.path.join("NN", arch, DEVICE, "TRAINING", target, folder)
    
    # Se la cartella non esiste (es. HW_SO_binary_out), skippa
    if not os.path.exists(attack_dir) or not os.path.exists(train_dir):
        return None

    def find_file(directory, suffix):
        if not os.path.exists(directory): return None
        for f in os.listdir(directory):
            if f.endswith(suffix):
                return os.path.join(directory, f)
        return None

    ge_path = find_file(attack_dir, "_ge.csv")
    if not ge_path: return None

    return {
        "ge": ge_path,
        "attack_time": find_file(attack_dir, "_attack_time.txt"),
        "attack_ram": find_file(attack_dir, "_attack_ram.txt"),
        "train_time": find_file(train_dir, "_training_time.txt"),
        "train_ram": find_file(train_dir, "_training_ram.txt"),
        "model": find_file(train_dir, ".h5") # Prende qualsiasi file .h5
    }

def read_ge(path, max_traces=MAX_TRACES):
    if path and os.path.exists(path): return np.loadtxt(path, delimiter=',')[:max_traces]
    return None

def get_break_point(ge_array, threshold=0.5):
    coords = np.where(ge_array <= threshold)[0]
    return coords[0] + 1 if len(coords) > 0 else MAX_TRACES

def read_metric_txt(path):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            for w in f.read().split():
                try: return float(w)
                except ValueError: continue
    return 0.0

def get_file_size_mb(path):
    if path and os.path.exists(path): return os.path.getsize(path) / (1024 * 1024)
    return 0.0

def extract_all_metrics(arch, target, out_mode, ptx):
    paths = get_all_paths(arch, target, out_mode, ptx)
    if not paths: return None # Skippa se la cartella o i file mancano
    
    ge = read_ge(paths["ge"])
    if ge is None: return None
    
    label_id = f"{arch}\n{target}\nOut: {out_mode}\nPTX: {ptx}"
    
    return {
        "id": label_id,
        "ge_path": paths["ge"],
        "traces": get_break_point(ge),
        "attack_time": read_metric_txt(paths["attack_time"]),
        "train_time": read_metric_txt(paths["train_time"]),
        "attack_ram": read_metric_txt(paths["attack_ram"]),
        "train_ram": read_metric_txt(paths["train_ram"]),
        "storage": get_file_size_mb(paths["model"])
    }

# ==========================================
# 4. NORMALIZZAZIONE COSTI E PARETO
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

def calculate_scores(models_data):
    if not models_data: return []
    
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
        
    return sorted(models_data, key=lambda x: x['area'])

def plot_radar_chart(models_data, title, out_path):
    if not models_data: return
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

    top_models = models_data[:3]
    for i, m in enumerate(top_models):
        values = m['radar_data']
        values += values[:1]
        label_text = f"{m['id'].replace(chr(10), ' ')} (Area: {m['area']:.3f})"
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=label_text, color=COLORS[i])
        ax.fill(angles, values, color=COLORS[i], alpha=0.15)

    plt.title(f"{title}\n(L'area MINORE indica prestazioni MIGLIORI)", size=15, y=1.12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==========================================
# 5. PLOTTING: BAR CHARTS & LINE CHARTS
# ==========================================
def plot_unified_bar(models_data, metric_key, ylabel, title, out_path, color):
    if not models_data: return
    labels = [m['id'] for m in models_data]
    vals = [m[metric_key] for m in models_data]

    # Dinamico: se ci sono 9 barre, allarghiamo un po' la figura per fare spazio ai nomi
    fig, ax = plt.subplots(figsize=(min(20, max(12, len(labels)*1.8)), 7))
    bars = ax.bar(labels, vals, color=color, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    for bar in bars:
        yval = bar.get_height()
        if 'traces' in metric_key and yval >= MAX_TRACES:
            ax.text(bar.get_x() + bar.get_width()/2, yval*1.02, f'>{MAX_TRACES}', ha='center', va='bottom', color='red', fontweight='bold')
        else:
            fmt = '{:.0f}' if 'traces' in metric_key else '{:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, yval*1.02, fmt.format(yval), ha='center', va='bottom', fontsize=10)

    if 'traces' in metric_key:
        ax.axhline(MAX_TRACES, color='red', linestyle='--', label=f'Max Limit ({MAX_TRACES})')
        ax.legend()
        ax.set_ylim(0, MAX_TRACES * 1.15)
    else:
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_ge_evolution_line(models_data, title, out_path):
    if not models_data: return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    valid_plots = 0
    
    for i, m in enumerate(models_data):
        ge = read_ge(m["ge_path"])
        if ge is not None:
            valid_plots += 1
            bp = m['traces']
            lbl_id = m['id'].replace(chr(10), ' ')
            lbl = f"{lbl_id} (Break: {bp})" if bp < MAX_TRACES else f"{lbl_id} (> {MAX_TRACES})"
            
            lw = 2.5 if bp < MAX_TRACES else 1.5
            alpha = 1.0 if bp < MAX_TRACES else 0.4
            
            ax.plot(np.arange(1, len(ge)+1), ge, color=COLORS[i % len(COLORS)], linewidth=lw, alpha=alpha, label=lbl)

    if valid_plots == 0:
        plt.close(fig)
        return

    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Break Threshold (GE < 0.5)')
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    
    ax.set_title(title)
    ax.set_xlabel("Number of Traces")
    ax.set_ylabel("Guessing Entropy")
    ax.set_ylim(-5, max(130, np.max(ge) + 10) if 'ge' in locals() and ge is not None else 130)
    
    # Legend spostata fuori per non coprire il grafico se ci sono 9 linee
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==========================================
# 6. ESECUZIONE PRINCIPALE
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print("Generazione Plot (Ricerca file dinamica - 9 Modelli)")
    print("========================================\n")
    
    best_overall_models = []

    for arch in ARCHS:
        print(f"Analizzando Architettura: {arch}...")
        arch_models = []
        
        for target in TARGETS:
            for out_mode in OUT_MODES:
                for ptx in PTX_MODES:
                    data = extract_all_metrics(arch, target, out_mode, ptx)
                    if data: arch_models.append(data)
                
        if not arch_models: continue
        
        print(f"-> Trovati {len(arch_models)} modelli validi per {arch}.")
        
        arch_models = calculate_scores(arch_models)
        best_overall_models.append(arch_models[0]) 
        
        out_traces = get_out_dir(arch, "TRACES")
        out_time = get_out_dir(arch, "TIME")
        out_ram = get_out_dir(arch, "RAM")
        out_storage = get_out_dir(arch, "STORAGE")
        out_pareto = get_out_dir(arch, "PARETO")

        plot_unified_bar(arch_models, "traces", "Minimum Traces (GE<0.5)", f"{arch} - Attack Efficiency", os.path.join(out_traces, f"{arch}_Traces_Bar.png"), '#377eb8')
        plot_ge_evolution_line(arch_models, f"Guessing Entropy Evolution - {arch}", os.path.join(out_traces, f"{arch}_GE_Evolution_Line.png"))
        plot_unified_bar(arch_models, "train_time", "Time (Seconds)", f"{arch} - Training Time", os.path.join(out_time, f"{arch}_TrainTime.png"), '#ff7f00')
        plot_unified_bar(arch_models, "attack_time", "Time (Seconds)", f"{arch} - Attack Time", os.path.join(out_time, f"{arch}_AttackTime.png"), '#e41a1c')
        plot_unified_bar(arch_models, "train_ram", "RAM (MB)", f"{arch} - Training Peak RAM", os.path.join(out_ram, f"{arch}_TrainRAM.png"), '#4daf4a')
        plot_unified_bar(arch_models, "attack_ram", "RAM (MB)", f"{arch} - Attack Peak RAM", os.path.join(out_ram, f"{arch}_AttackRAM.png"), '#984ea3')
        plot_unified_bar(arch_models, "storage", "File Size (MB)", f"{arch} - Model Storage Size", os.path.join(out_storage, f"{arch}_Storage.png"), '#a65628')
        plot_radar_chart(arch_models, f"5D Pareto Analysis: Top {arch} Models", os.path.join(out_pareto, f"{arch}_Radar_Pareto.png"))


    if best_overall_models:
        print("\nGenerazione Comparison dei Best Models...")
        best_overall_models = calculate_scores(best_overall_models)
        comp_dir = get_out_dir("Architecture_Comparisons")
        
        for sub in ["TRACES", "TIME", "RAM", "STORAGE", "PARETO"]:
            os.makedirs(os.path.join(comp_dir, sub), exist_ok=True)

        plot_unified_bar(best_overall_models, "traces", "Minimum Traces (GE<0.5)", "BEST MODELS - Attack Efficiency", os.path.join(comp_dir, "TRACES", "All_Traces_Bar.png"), '#377eb8')
        plot_ge_evolution_line(best_overall_models, "BEST MODELS - Guessing Entropy Evolution", os.path.join(comp_dir, "TRACES", "All_GE_Evolution_Line.png"))
        plot_unified_bar(best_overall_models, "train_time", "Time (Seconds)", "BEST MODELS - Training Time", os.path.join(comp_dir, "TIME", "All_TrainTime.png"), '#ff7f00')
        plot_unified_bar(best_overall_models, "attack_time", "Time (Seconds)", "BEST MODELS - Attack Time", os.path.join(comp_dir, "TIME", "All_AttackTime.png"), '#e41a1c')
        plot_unified_bar(best_overall_models, "train_ram", "RAM (MB)", "BEST MODELS - Training RAM", os.path.join(comp_dir, "RAM", "All_TrainRAM.png"), '#4daf4a')
        plot_unified_bar(best_overall_models, "attack_ram", "RAM (MB)", "BEST MODELS - Attack RAM", os.path.join(comp_dir, "RAM", "All_AttackRAM.png"), '#984ea3')
        plot_unified_bar(best_overall_models, "storage", "File Size (MB)", "BEST MODELS - Storage Size", os.path.join(comp_dir, "STORAGE", "All_Storage.png"), '#a65628')
        
        plot_radar_chart(best_overall_models, "Ultimate 5D Pareto: Best Models Comparison", os.path.join(comp_dir, "PARETO", "Ultimate_Radar_Pareto.png"))

    print("\nTutto generato con successo! Ora ci sono 9 barre per grafico.")
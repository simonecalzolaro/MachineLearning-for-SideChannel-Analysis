import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

# ==========================================
# 1. SETUP PATHS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, "./utils")

try:
    from constants import BTA_METRICS_PATH_RISCURE_PINATA, PATH_PLOTS_RISCURE_PINATA, MAX_TRACES
except ImportError:
    BTA_METRICS_PATH_RISCURE_PINATA = "BTA/metrics/riscure_pinata"
    PATH_PLOTS_RISCURE_PINATA = "plots/riscure_pinata/"
    MAX_TRACES = 500

DEVICE = "riscure_pinata"
OUT_DIR = os.path.join(PATH_PLOTS_RISCURE_PINATA, "Ultimate_Showdown")
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetics
COLORS = {'BTA': '#2ca02c', 'MLP': '#e41a1c', 'CNN_ZAID': '#377eb8'}
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 12, 'figure.figsize': (10, 6), 
    'axes.grid': True, 'grid.alpha': 0.4, 'grid.linestyle': '--'
})

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def read_ge(path, max_traces=MAX_TRACES):
    if path and os.path.exists(path): 
        return np.loadtxt(path, delimiter=',')[:max_traces]
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

def get_file_size_kb(path):
    if path and os.path.exists(path): 
        return os.path.getsize(path) / 1024.0
    return 0.0

def normalize_cost(values):
    vals = np.array(values, dtype=float)
    if len(vals) == 0: return vals
    min_v, max_v = np.min(vals), np.max(vals)
    
    if max_v == min_v: 
        norm_vals = np.ones_like(vals) if max_v > 0 else np.zeros_like(vals)
    else:
        norm_vals = (vals - min_v) / (max_v - min_v)
        
    # AGGIUNTA FLOOR: comprime l'intervallo da [0.0, 1.0] a [0.1, 1.0]
    # In questo modo il modello "perfetto" non scompare al centro (0.0) nel radar
    return 0.1 + (norm_vals * 0.9)

def calculate_polygon_area(values):
    N = len(values)
    angle = 2 * pi / N
    area = sum(0.5 * values[i] * values[(i + 1) % N] * np.sin(angle) for i in range(N))
    max_area = (N / 2) * np.sin(angle)
    return area / max_area

def get_champions_by_traces(models_data):
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
        
    return sorted(models_data, key=lambda x: (x['traces'], x['area']))

# ==========================================
# 3. DATA EXTRACTION
# ==========================================
def extract_bta_models():
    features_choice = ['PCA_', 'LDA_', 'PCA_LDA_', 'PCA_HW_', 'LDA_HW_', 'PCA_LDA_HW_']
    operations = ['SUM']
    models = []
    
    for f_c in features_choice:
        for op in operations:
            filepath = os.path.join(BTA_METRICS_PATH_RISCURE_PINATA, f"{f_c}{op}_metrics.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    d = json.load(f)
                    
                    # Fix automatico se la RAM del BTA è in bytes
                    a_ram = d['Attack_RAM_usage']
                    t_ram = d['Training_RAM_usage']
                    if a_ram > 10000: a_ram /= (1024.0 * 1024.0)
                    if t_ram > 10000: t_ram /= (1024.0 * 1024.0)
                    
                    models.append({
                        "id": f"BTA ({f_c.replace('_', '+').strip('+')})",
                        "family": "BTA",
                        "traces": min(d['GE_crossing_point'], MAX_TRACES),
                        "attack_time": d['Attack_time'],
                        "train_time": d['Training_time'],
                        "attack_ram": a_ram,
                        "train_ram": t_ram,
                        "storage": d['Storage_size'] / 1024.0,
                        "ge_curve": d.get('GE_curve', []) # Estraggo la curva GE dal JSON
                    })
    return models

def extract_nn_models(arch):
    models = []
    base_dir = Path("NN") / arch / DEVICE
    if not base_dir.exists(): return models

    for ge_path in base_dir.rglob("*_ge.csv"):
        config_name = ge_path.name.replace("_ge.csv", "")
        train_dir = Path(str(ge_path.parent).replace("ATTACK", "TRAINING"))
        attack_dir = ge_path.parent
        
        # Ricerca SUPER ROBUSTA ignorando maiuscole/minuscole
        def get_val(directory, keyword):
            if not directory.exists(): return 0.0
            for f in directory.rglob("*.txt"):
                if keyword.lower() in f.name.lower():
                    return read_metric_txt(str(f))
            return 0.0

        def get_size(directory, ext):
            if not directory.exists(): return 0.0
            for f in directory.rglob(f"*{ext}"):
                return get_file_size_kb(str(f))
            return 0.0

        attack_time = get_val(attack_dir, "attack_time")
        attack_ram = get_val(attack_dir, "attack_ram")
        train_time = get_val(train_dir, "training_time")
        train_ram = get_val(train_dir, "training_ram")
        storage = get_size(train_dir, ".h5")

        ge = read_ge(str(ge_path))
        if ge is None: continue

        target_str = str(ge_path).upper()
        target = "SBOX_OUT" if "SBOX" in target_str else "HW_SO" if "HW" in target_str else "UNK"
        out_mode = "cat" if "categorical" in config_name else "bin"
        ptx_mode = "none" if "none" in config_name else "scalar" if "scalar" in config_name else "bin" if "binary" in config_name else "unk"
        
        lbl = f"{arch} ({target}, {out_mode}, ptx:{ptx_mode})"

        models.append({
            "id": lbl,
            "family": arch,
            "traces": get_break_point(ge),
            "attack_time": attack_time,
            "train_time": train_time,
            "attack_ram": attack_ram,
            "train_ram": train_ram,
            "storage": storage,
            "ge_curve": ge # Estraggo l'array GE dal CSV
        })
    return models

# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
def plot_ge_evolution_line(models, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    valid_plots = 0
    
    for m in models:
        ge = m.get('ge_curve', [])
        if len(ge) > 0:
            valid_plots += 1
            bp = m['traces']
            lbl_id = m['family']
            lbl = f"{lbl_id} Champion (Break: {bp})" if bp < MAX_TRACES else f"{lbl_id} Champion (> {MAX_TRACES})"
            
            # Taglia l'array fino a MAX_TRACES per coerenza visiva
            ge_plot = ge[:MAX_TRACES]
            
            ax.plot(np.arange(1, len(ge_plot)+1), ge_plot, color=COLORS[m['family']], linewidth=2.5, label=lbl)

    if valid_plots == 0:
        plt.close(fig)
        return

    ax.axhline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Success Threshold (GE < 0.5)')
    ax.axhline(0, color='gray', linewidth=1, alpha=0.3)
    
    ax.set_title(title, fontweight='bold', size=15)
    ax.set_xlabel("Number of Traces", fontweight='bold')
    ax.set_ylabel("Guessing Entropy", fontweight='bold')
    
    # Imposta dinamicamente il limite Y per non tagliare le curve
    max_y = 130
    for m in models:
        ge = m.get('ge_curve', [])
        if len(ge) > 0:
            max_y = max(max_y, np.max(ge[:MAX_TRACES]) + 10)
    ax.set_ylim(-5, max_y)
    ax.set_xlim(0, MAX_TRACES)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close(fig)

def plot_comparison_bar(models, metric_key, ylabel, title, filename, log_scale=False):
    labels = [m['family'] for m in models]
    vals = [m[metric_key] for m in models]
    bar_colors = [COLORS[fam] for fam in labels]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', size=15)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=min(vals)*0.1 if min(vals)>0 else 0.1, top=max(vals)*5)
    else:
        ax.set_ylim(0, max(vals) * 1.15)
        
    for bar, val in zip(bars, vals):
        yval = bar.get_height()
        fmt = '{:.0f}' if 'traces' in metric_key else '{:.1f}'
        text_val = f'>{MAX_TRACES}' if ('traces' in metric_key and val >= MAX_TRACES) else fmt.format(val)
        
        idx = [b.get_x() for b in bars].index(bar.get_x())
        model_id = models[idx]['id']
        
        ax.text(bar.get_x() + bar.get_width()/2, yval * (1.1 if log_scale else 1.02), 
                text_val, ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.text(bar.get_x() + bar.get_width()/2, yval * (0.5 if log_scale else 0.5), 
                model_id.replace(" (", "\n("), ha='center', va='center', fontweight='bold', fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close(fig)

def plot_ultimate_radar(models):
    traces = [m['traces'] for m in models]
    a_times = [m['attack_time'] for m in models]
    t_times = [m['train_time'] for m in models]
    max_rams = [max(m['attack_ram'], m['train_ram']) for m in models]
    storages = [m['storage'] for m in models]

    c_traces = normalize_cost(traces)
    c_atime = normalize_cost(a_times)
    c_ttime = normalize_cost(t_times)
    c_ram = normalize_cost(max_rams)
    c_storage = normalize_cost(storages)

    categories = ['Traces\n(GE<0.5)', 'Attack Time', 'Train Time', 'Peak RAM', 'Storage']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", "Worst\n(Max Cost)"], color="red", size=10, alpha=0.7)
    ax.text(0, 0, 'Best\n(Min Cost)', horizontalalignment='center', verticalalignment='center', size=11, color='green', fontweight='bold')
    plt.ylim(0, 1.1)

    for i, m in enumerate(models):
        radar_data = [c_traces[i], c_atime[i], c_ttime[i], c_ram[i], c_storage[i]]
        area = calculate_polygon_area(radar_data)
        radar_data.append(radar_data[0])
        
        label_text = f"{m['family']} Champion (Traces: {m['traces']}, Area: {area:.2f})"
        c = COLORS[m['family']]
        
        ax.plot(angles, radar_data, linewidth=3.0, linestyle='solid', label=label_text, color=c)
        ax.fill(angles, radar_data, color=c, alpha=0.15)

    plt.title("Ultimate Showdown: 5D Pareto Comparison\n(Ranked internally across Best Trace Models)", size=16, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Ultimate_Showdown_Radar.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[*] Extracting models...")
    all_bta = extract_bta_models()
    all_mlp = extract_nn_models("MLP")
    all_cnn = extract_nn_models("CNN_ZAID")

    best_bta = get_champions_by_traces(all_bta)[0] if all_bta else None
    best_mlp = get_champions_by_traces(all_mlp)[0] if all_mlp else None
    best_cnn = get_champions_by_traces(all_cnn)[0] if all_cnn else None

    champions = [m for m in [best_bta, best_mlp, best_cnn] if m is not None]

    for c in champions:
        c['max_ram'] = max(c['train_ram'], c['attack_ram'])

    print("\n[*] SANITY CHECK DEI DATI ESTRATTI:")
    for c in champions:
        print(f"    -> {c['family']:>8}: Traces={c['traces']:>3}, Time={c['train_time']:>7.1f}s, PeakRAM={c['max_ram']:>6.1f}MB, Storage={c['storage']:>7.1f}KB")

    print("\n[*] Generating Comparison Plots...")
    
    # Aggiunto il grafico a linee della GE
    plot_ge_evolution_line(champions, "Guessing Entropy Evolution (Champions)", "Showdown_GE_Evolution.png")
    
    plot_comparison_bar(champions, "traces", "Minimum Traces (GE<0.5)", "Data Complexity Comparison", "Showdown_Traces.png")
    plot_comparison_bar(champions, "train_time", "Time (Seconds, Log Scale)", "Training Time Comparison", "Showdown_TrainTime.png", log_scale=True)
    plot_comparison_bar(champions, "attack_time", "Time (Seconds)", "Attack (Inference) Time Comparison", "Showdown_AttackTime.png")
    plot_comparison_bar(champions, "max_ram", "RAM (MB, Log Scale)", "Peak RAM Usage Comparison", "Showdown_PeakRAM.png", log_scale=True)
    plot_comparison_bar(champions, "storage", "File Size (KB, Log Scale)", "Model Storage Size Comparison", "Showdown_Storage.png", log_scale=True)

    plot_ultimate_radar(champions)
    print(f"[*] All plots saved successfully in: {OUT_DIR}")
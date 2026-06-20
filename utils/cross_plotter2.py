import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

# ==========================================
# 1. SETUP PATHS & DEVICE ARGUMENT
# ==========================================
if len(sys.argv) < 2:
    print("[!] Errore: Devi specificare il target.")
    print("Uso: python3 utils/cross_plotter.py [atmega | riscure_pinata]")
    sys.exit(1)

DEVICE = sys.argv[1]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, "./utils")

try:
    from constants import MAX_TRACES
    if DEVICE == "atmega":
        from constants import BTA_METRICS_PATH_ATMEGA as BTA_METRICS_PATH
        from constants import PATH_PLOTS_ATMEGA as PATH_PLOTS
    else:
        from constants import BTA_METRICS_PATH_RISCURE_PINATA as BTA_METRICS_PATH
        from constants import PATH_PLOTS_RISCURE_PINATA as PATH_PLOTS
except ImportError:
    BTA_METRICS_PATH = f"BTA/metrics/{DEVICE}"
    PATH_PLOTS = f"plots/{DEVICE}/"
    MAX_TRACES = 500

OUT_DIR = os.path.join(PATH_PLOTS, "Ultimate_Showdown")
os.makedirs(OUT_DIR, exist_ok=True)

# Limiti per l'ATmega (Convergenze lente)
PRACTICAL_LIMIT = MAX_TRACES
ANALYTICAL_HORIZON = 10000

# Aesthetics (Aggiunto MLP_PRE in Viola)
COLORS = {'BTA': '#2ca02c', 'MLP': '#e41a1c', 'MLP_PRE': '#984ea3', 'CNN_ZAID': '#377eb8'}
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 11, 'figure.figsize': (11, 7), 
    'axes.grid': True, 'grid.alpha': 0.4, 'grid.linestyle': '--'
})

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def read_ge(path, max_traces=ANALYTICAL_HORIZON):
    if path and os.path.exists(path): 
        return np.loadtxt(path, delimiter=',')[:max_traces]
    return None

def get_break_point(ge_array, threshold=0.5):
    coords = np.where(ge_array <= threshold)[0]
    return coords[0] + 1 if len(coords) > 0 else ANALYTICAL_HORIZON

def read_metric_txt(path):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            for w in f.read().split():
                try: return float(w)
                except ValueError: continue
    return 0.0

def get_file_size_mb(path):
    if path and os.path.exists(path): 
        return os.path.getsize(path) / (1024.0 * 1024.0)
    return 0.0

def normalize_cost(values):
    vals = np.array(values, dtype=float)
    if len(vals) == 0: return vals
    min_v, max_v = np.min(vals), np.max(vals)
    
    if max_v == min_v: 
        norm_vals = np.ones_like(vals) if max_v > 0 else np.zeros_like(vals)
    else:
        norm_vals = (vals - min_v) / (max_v - min_v)
        
    return 0.1 + (norm_vals * 0.9)

def calculate_polygon_area(values):
    w = [0.60, 0.1, 0.1, 0.1, 0.1]
    N = len(values)
    angle = 2 * pi / N
    
    weighted_area = sum(w[i] * (0.5 * values[i] * values[(i + 1) % N] * np.sin(angle)) for i in range(N))
    max_weighted_area = sum(w[i] * (0.5 * 1.0 * 1.0 * np.sin(angle)) for i in range(N))
    
    return weighted_area / max_weighted_area

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
            filepath = os.path.join(BTA_METRICS_PATH, f"{f_c}{op}_metrics.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    d = json.load(f)
                    
                    a_ram = d['Attack_RAM_usage']
                    t_ram = d['Training_RAM_usage']
                    if a_ram > 10000: a_ram /= (1024.0 * 1024.0)
                    if t_ram > 10000: t_ram /= (1024.0 * 1024.0)
                    
                    cp = d['GE_crossing_point']
                    if cp < 0 or cp > ANALYTICAL_HORIZON: cp = ANALYTICAL_HORIZON

                    models.append({
                        "id": f"BTA ({f_c.replace('_', '+').strip('+')})",
                        "family": "BTA",
                        "traces": cp,
                        "attack_time": d['Attack_time'],
                        "train_time": d['Training_time'],
                        "attack_ram": a_ram,
                        "train_ram": t_ram,
                        "storage": d['Storage_size'] / (1024.0 * 1024.0),
                        "ge_curve": d.get('GE_curve', [])
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
        
        def get_val(directory, keyword):
            if not directory.exists(): return 0.0
            for f in directory.rglob("*.txt"):
                if keyword.lower() in f.name.lower():
                    return read_metric_txt(str(f))
            return 0.0

        def get_size(directory, ext):
            if not directory.exists(): return 0.0
            for f in directory.rglob(f"*{ext}"):
                return get_file_size_mb(str(f))
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
            "ge_curve": ge 
        })
    return models

# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
def plot_ge_evolution_line(models, title, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    valid_plots = 0
    max_x = PRACTICAL_LIMIT
    
    for m in models:
        ge = m.get('ge_curve', [])
        if len(ge) > 0:
            valid_plots += 1
            bp = m['traces']
            lbl_id = m['family']
            
            if bp < ANALYTICAL_HORIZON:
                lbl = f"{lbl_id} Champion (Break: {bp})"
                max_x = max(max_x, bp + 500)
                alpha = 1.0
                lw = 2.5
            else:
                lbl = f"{lbl_id} Champion (Not Breaked)"
                max_x = max(max_x, len(ge))
                alpha = 0.5
                lw = 1.5
            
            ge_plot = ge[:ANALYTICAL_HORIZON]
            ax.plot(np.arange(1, len(ge_plot)+1), ge_plot, color=COLORS[m['family']], linewidth=lw, alpha=alpha, label=lbl)

    if valid_plots == 0:
        plt.close(fig)
        return

    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Success Threshold (GE < 0.5)')
    ax.axvline(PRACTICAL_LIMIT, color='gray', linestyle=':', linewidth=2, label=f'Practical Auditing Limit ({PRACTICAL_LIMIT})')
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    
    ax.set_title(title, fontweight='bold', size=15)
    ax.set_xlabel("Number of Traces", fontweight='bold')
    ax.set_ylabel("Guessing Entropy", fontweight='bold')
    
    ax.set_ylim(-5, 130)
    ax.set_xlim(0, max_x)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close(fig)

def plot_comparison_bar(models, metric_key, ylabel, title, filename, log_scale=False):
    labels = [m['family'] for m in models]
    vals = [m[metric_key] for m in models]
    bar_colors = [COLORS[fam] for fam in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', size=15)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=min(vals)*0.1 if min(vals)>0 else 0.1, top=max(vals)*5)
    else:
        max_limit = max(vals) * 1.15
        if 'traces' in metric_key:
            max_limit = max(max_limit, PRACTICAL_LIMIT * 1.5)
        ax.set_ylim(0, max_limit)
        
    for bar, val in zip(bars, vals):
        yval = bar.get_height()
        fmt = '{:.0f}' if 'traces' in metric_key else '{:.2f}'
        
        if 'traces' in metric_key and val >= ANALYTICAL_HORIZON:
            text_val = "Not\nBreaked"
            color_val = 'red'
        else:
            text_val = fmt.format(val)
            color_val = 'black'
        
        idx = [b.get_x() for b in bars].index(bar.get_x())
        model_id = models[idx]['id']
        
        ax.text(bar.get_x() + bar.get_width()/2, yval * (1.1 if log_scale else 1.02), 
                text_val, ha='center', va='bottom', fontweight='bold', fontsize=11, color=color_val)
        
        ax.text(bar.get_x() + bar.get_width()/2, yval * (0.5 if log_scale else 0.5), 
                model_id.replace(" (", "\n("), ha='center', va='center', fontweight='bold', fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    if 'traces' in metric_key:
        ax.axhline(PRACTICAL_LIMIT, color='red', linestyle='--', linewidth=2, label=f'Practical Auditing Limit ({PRACTICAL_LIMIT})')
        ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close(fig)

def plot_ultimate_radar(models):
    # Rimuovi i modelli che non hanno fatto breccia per non sballare il Pareto
    valid_models = [m for m in models if m['traces'] < ANALYTICAL_HORIZON]
    
    if not valid_models:
        print("[!] Nessun modello valido per il Pareto Radar finale.")
        return

    traces = [m['traces'] for m in valid_models]
    a_times = [m['attack_time'] for m in valid_models]
    t_times = [m['train_time'] for m in valid_models]
    max_rams = [max(m['attack_ram'], m['train_ram']) for m in valid_models]
    storages = [m['storage'] for m in valid_models]

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

    for i, m in enumerate(valid_models):
        radar_data = [c_traces[i], c_atime[i], c_ttime[i], c_ram[i], c_storage[i]]
        area = calculate_polygon_area(radar_data)
        radar_data.append(radar_data[0])
        
        label_text = f"{m['family']} (Area: {area:.3f})"
        c = COLORS[m['family']]
        
        ax.plot(angles, radar_data, linewidth=3.0, linestyle='solid', label=label_text, color=c)
        ax.fill(angles, radar_data, color=c, alpha=0.15)

    plt.title(f"Ultimate Showdown: 5D Pareto Comparison ({DEVICE.upper()})\n(Ranked internally across Best Trace Models)", size=16, y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Ultimate_Showdown_Radar.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"[*] Extracting models for target: {DEVICE.upper()}...")
    all_bta = extract_bta_models()
    all_mlp = extract_nn_models("MLP")
    all_mlp_pre = extract_nn_models("MLP_PRE")
    all_cnn = extract_nn_models("CNN_ZAID")

    best_bta = get_champions_by_traces(all_bta)[0] if all_bta else None
    best_mlp = get_champions_by_traces(all_mlp)[0] if all_mlp else None
    best_mlp_pre = get_champions_by_traces(all_mlp_pre)[0] if all_mlp_pre else None
    best_cnn = get_champions_by_traces(all_cnn)[0] if all_cnn else None

    # Ora ci sono 4 campioni nel ring
    champions = [m for m in [best_bta, best_mlp, best_mlp_pre, best_cnn] if m is not None]

    for c in champions:
        c['max_ram'] = max(c['train_ram'], c['attack_ram'])

    print("\n[*] SANITY CHECK DEI DATI ESTRATTI:")
    for c in champions:
        print(f"    -> {c['family']:>8}: Traces={c['traces']:>5}, Time={c['train_time']:>7.1f}s, PeakRAM={c['max_ram']:>6.1f}MB, Storage={c['storage']:>7.2f}MB")

    print("\n[*] Generating Comparison Plots...")
    
    plot_ge_evolution_line(champions, f"Guessing Entropy Evolution (Champions - {DEVICE.upper()})", "Showdown_GE_Evolution.png")
    plot_comparison_bar(champions, "traces", "Minimum Traces (GE<0.5)", f"Data Complexity Comparison ({DEVICE.upper()})", "Showdown_Traces.png")
    plot_comparison_bar(champions, "train_time", "Time (Seconds, Log Scale)", f"Training Time Comparison ({DEVICE.upper()})", "Showdown_TrainTime.png", log_scale=True)
    plot_comparison_bar(champions, "attack_time", "Time (Seconds)", f"Attack (Inference) Time Comparison ({DEVICE.upper()})", "Showdown_AttackTime.png")
    plot_comparison_bar(champions, "max_ram", "RAM (MB, Log Scale)", f"Peak RAM Usage Comparison ({DEVICE.upper()})", "Showdown_PeakRAM.png", log_scale=True)
    plot_comparison_bar(champions, "storage", "File Size (MB, Log Scale)", f"Model Storage Size Comparison ({DEVICE.upper()})", "Showdown_Storage.png", log_scale=True)

    plot_ultimate_radar(champions)
    print(f"[*] All plots saved successfully in: {OUT_DIR}")
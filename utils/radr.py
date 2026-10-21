import os
import math
import csv
from pathlib import Path

# Constant trigonometric sine value for 72 degrees (pentagon sector angle)
sin_72 = math.sin(math.radians(72))

# Universal reference bounds for physical metrics normalization (10.0 to 10000.0 for traces)
UNIVERSAL_REFERENCE_BOUNDS = {
    "Traces":     [10.0, 10000.0],    # Logarithmic scale mapping
    "TrainTime":  [0.0, 25000.0],     # Linear scale mapping (seconds)
    "AttackTime": [0.0, 30.0],        # Linear scale mapping (seconds)
    "RAM":        [0.0, 4500.0],      # Linear scale mapping (MB)
    "Storage":    [0.0, 4500.0]       # Linear scale mapping (KB)
}

# Asymmetric Pareto weight vector: 60% assigned to Data Complexity (Traces)
SECTOR_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]

# Format: [Traces, TrainTime (s), AttackTime (s), PeakRAM (MB), Storage (KB)]
CUSTOM_MLP_MEASUREMENTS = {
    "HW_SO Out:Cat Ptx:Bin":     [10000.0, 481.2,  14.9, 3222.0, 137.0],
    "HW_SO Out:Cat Ptx:None":    [13.0,    1336.8, 15.4, 3249.3, 271.9],
    "HW_SO Out:Cat Ptx:Scal":    [18.0,    2296.9, 15.0, 3252.0, 208.2],
    "SBOX_OUT Out:Bin Ptx:Bin":  [10000.0, 653.0,  21.2, 3139.6, 220.0],
    "SBOX_OUT Out:Bin Ptx:None": [27.0,    1158.3, 20.7, 3232.5, 212.0],
    "SBOX_OUT Out:Bin Ptx:Scal": [10000.0, 1187.5, 21.4, 3114.1, 222.0],
    "SBOX_OUT Out:Cat Ptx:Bin":  [10000.0, 212.8,  21.1, 3320.3, 145.0],
    "SBOX_OUT Out:Cat Ptx:None": [10.0,    2645.0, 20.9, 3578.9, 211.0],
    "SBOX_OUT Out:Cat Ptx:Scal": [10000.0, 452.2,  20.6, 3147.3, 378.0]
}

def normalize_value(val, key):
    """
    Applies hybrid normalization: Logarithmic Base-10 scaling for Data Complexity (Traces)
    and Linear scaling for hardware/temporal overhead metrics.
    """
    b_min, b_max = UNIVERSAL_REFERENCE_BOUNDS[key]
    val = max(b_min, min(b_max, val))  # Safety clipping within reference bounds

    if key == "Traces":
        log_val = math.log10(val)
        log_min = math.log10(b_min)
        log_max = math.log10(b_max)
        return 0.1 + 0.9 * (log_val - log_min) / (log_max - log_min)
    else:
        return 0.1 + 0.9 * (val - b_min) / (b_max - b_min)

def calculate_omega(r, w):
    """
    Calculates the aggregate weighted geometric inefficiency metric (Omega)
    by summing the areas of 5 adjacent triangular polars.
    """
    total_area = 0.0
    for i in range(5):
        r_curr = r[i]
        r_next = r[(i + 1) % 5]
        sector_area = 0.5 * r_curr * r_next * sin_72
        total_area += w[i] * sector_area
    return total_area

def main():
    print("=" * 85)
    print("   CUSTOM MLP ABLATION STUDY - PARETO OMEGA & TIKZ GENERATOR")
    print("   [!] Logarithmic Scaling Enabled for Data Complexity (Traces)")
    print("=" * 85)

    color_idx = 1
    
    for model_name, raw_values in CUSTOM_MLP_MEASUREMENTS.items():
        traces, train_time, attack_time, ram, storage = raw_values
        
        # Calculate normalized radius vector r_i
        r = [
            normalize_value(traces, "Traces"),
            normalize_value(train_time, "TrainTime"),
            normalize_value(attack_time, "AttackTime"),
            normalize_value(ram, "RAM"),
            normalize_value(storage, "Storage")
        ]
        
        # Calculate scalar inefficiency metric
        omega = calculate_omega(r, SECTOR_WEIGHTS)
        
        print(f"\nModello: {model_name}")
        print(f"  Raw Metrics: Traces={traces}, TrainTime={train_time:.1f}s, AttackTime={attack_time:.1f}s, RAM={ram:.1f}MB, Storage={storage:.1f}KB")
        print(f"  Vector r_i:  {[round(x, 4) for x in r]}")
        print(f"  Omega (Ω):   {omega:.4f}")
        
        # Generate TikZ polar coordinates for 5 axes (90, 18, 306, 234, 162 degrees)
        tikz_coords = []
        for i in range(5):
            angle = (90 - i * 72) % 360
            scaled_r = r[i] * 4.0
            tikz_coords.append(f"({angle}:{scaled_r:.2f})")
            
        clean_model_tag = model_name.replace(":", "").replace(" ", "_").replace("-", "_")
        print(f"  TikZ Polygon Code:")
        print(f"    \\draw[color{color_idx}, very thick, fill=color{color_idx}, fill opacity=0.08] " + " -- ".join(tikz_coords) + " -- cycle;")
        
        color_idx += 1

    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
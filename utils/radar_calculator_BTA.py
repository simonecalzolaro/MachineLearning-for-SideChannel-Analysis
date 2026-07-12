import math

# 1. Definizione dell'angolo costante tra gli assi (72 gradi in radianti)
theta_rad = math.radians(72)
sin_72 = math.sin(theta_rad)

# 2. Inserimento dei dati reali estratti dai tuoi JSON (Fronte Riscure Piñata)
# Formato: [Traces, TrainingTime, AttackTime, PeakRAM, Storage]
raw_data = {
    "PCA":        [1647, 0.8730417949846014,22.518840874615126,6.05390625,220.0],
    #LDA":        [11, 8.85230990019627,19.49929721080698,1591.14921875,144.0],
    "PCA_LDA":    [6257, 1.1700923322001473,18.400790582573972,2635.27421875,144.0],
    "PCA_HW":     [4353, 0.5557052950141952,18.21278005943168,2607.04296875,7.73],
    #"LDA_HW":     [22, 8.157814306207001,17.933392972801812,4183.375,5.06],
    "PCA_LDA_HW": [9230, 0.7189348165877163,17.837776283966377,2646.64453125,5.06]
}

# 3. Funzione per la normalizzazione lineare nell'intervallo [0.1, 1.0]
def normalize_metric(values):
    min_v = min(values)
    max_v = max(values)
    # Se i valori sono tutti uguali, evita la divisione per zero mappando a 0.1
    if max_v == min_v:
        return [0.1 for _ in values]
    return [0.1 + 0.9 * (v - min_v) / (max_v - min_v) for v in values]

# Estraiamo le colonne per normalizzarle in modo indipendente
metrics_keys = ["Traces", "TrainTime", "AttackTime", "RAM", "Storage"]
normalized_by_metric = {}

for col_idx, metric_name in enumerate(metrics_keys):
    col_values = [raw_data[model][col_idx] for model in raw_data]
    normalized_by_metric[metric_name] = normalize_metric(col_values)

# Ricostruiamo il dizionario dei vettori normalizzati per ciascun modello
normalized_models = {}
for row_idx, model in enumerate(raw_data.keys()):
    normalized_models[model] = [
        normalized_by_metric["Traces"][row_idx],
        normalized_by_metric["TrainTime"][row_idx],
        normalized_by_metric["AttackTime"][row_idx],
        normalized_by_metric["RAM"][row_idx],
        normalized_by_metric["Storage"][row_idx]
    ]

# 4. Funzione per calcolare l'area pesata Omega
def calculate_omega(r, w):
    # r è la lista dei 5 raggi normalizzati del modello [r_1, r_2, r_3, r_4, r_5]
    # w è il vettore dei pesi dei settori [w_1, w_2, w_3, w_4, w_5]
    total_area = 0.0
    for i in range(5):
        r_curr = r[i]
        r_next = r[(i + 1) % 5]  # L'indice 5 fa il wrap-around all'indice 0 (ciclico)
        
        # Area geometrica del singolo settore triangolare
        sector_area = 0.5 * r_curr * r_next * sin_72
        # Moltiplicazione per il rispettivo peso asimmetrico
        weighted_sector = w[i] * sector_area
        total_area += weighted_sector
    return total_area

# Pesi asimmetrici richiesti: 60% per il settore primario e 10% per i restanti 4
sector_weights = [0.6, 0.1, 0.1, 0.1, 0.1]

# 5. Output dei risultati e codice LaTeX per i poligoni
print("--- VETTORI NORMALIZZATI E VALORI DI INEFFICIENZA GLOBALE ---")
for model, r in normalized_models.items():
    omega = calculate_omega(r, sector_weights)
    print(f"\nModello: {model}")
    print(f"  Vettore r_i: {['{:.3f}'.format(val) for val in r]}")
    print(f"  Area Pesata \Omega: {omega:.4f}")
    # Genera automaticamente la stringa TikZ pronta per Overleaf
    tikz_points = " -- ".join([f"({90 - i*72}:{r[i]:.2f}*4)" for i in range(5)])
    print(f"  Codice TikZ: \\draw[color{model}, very thick, fill=color{model}, fill opacity=0.15] {tikz_points} -- cycle;")
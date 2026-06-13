import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
import sys
sys.path.insert(0, '../utils')
from constants import sbox_atmega_flat,PATH_TRACES_ATMEGA,PATH_PLOTS_ATMEGA,BYTE_IDX
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change the Current Working Directory to the project root
os.chdir(PROJECT_ROOT)


# La S-Box dell'AES
AES_Sbox = sbox_atmega_flat

# Look-up table per calcolare il Peso di Hamming (numero di bit a 1)
HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

def main():
    print(f"Caricamento dati da {PATH_TRACES_ATMEGA}...")
    data = np.load(PATH_TRACES_ATMEGA)
    
    # Usiamo solo le tracce di training per trovare il delta (50.000 sono più che sufficienti)
    traces = data['X_train']
    ptx = data['ptxs_train']
    key = data['keys_train']
    masks = data['masks_train']
    
    num_traces = traces.shape[0]
    num_samples = traces.shape[1]
    print(f"Tracce caricate: {num_traces}, Campioni per traccia: {num_samples}")

    print("\nCalcolo dei valori intermedi (Leakage Models)...")
    # 1. Valore della Maschera (M)
    M = masks[:, BYTE_IDX]
    
    # 2. Uscita S-Box mascherata (V ^ M)
    # Valore non mascherato V = Sbox(P ^ K)
    V = AES_Sbox[ptx[:, BYTE_IDX] ^ key[:, BYTE_IDX]]
    V_masked = V ^ M
    
    # 3. Conversione in Peso di Hamming (HW)
    # Il consumo elettrico è proporzionale al numero di bit a 1 elaborati nel registro
    HW_M = HW[M]
    HW_V_masked = HW[V_masked]

    print("Inizio calcolo Correlazione di Pearson (CPA)...")
    corr_M = np.zeros(num_samples)
    corr_V_masked = np.zeros(num_samples)
    
    start_time = time.time()
    
    # Calcoliamo la correlazione per ogni istante temporale (colonna)
    for t in range(num_samples):
        # Traccia il progresso
        if t % 100 == 0 and t > 0:
            print(f" -> Processati {t}/{num_samples} campioni...")
            
        col = traces[:, t]
        
        # Correlazione con la Maschera
        corr_M[t], _ = pearsonr(col, HW_M)
        # Correlazione con il Valore Mascherato
        corr_V_masked[t], _ = pearsonr(col, HW_V_masked)

    print(f"Correlazione completata in {time.time() - start_time:.2f} secondi.")

    # Troviamo gli indici dei picchi massimi (in valore assoluto)
    t1 = np.argmax(np.abs(corr_M))
    t2 = np.argmax(np.abs(corr_V_masked))
    delta = abs(t1 - t2)

    print("\n" + "="*50)
    print(" RISULTATI DELL'ANALISI DEL LEAKAGE")
    print("="*50)
    print(f"Picco della Maschera (t1):        Campione {t1}  (Correlazione: {abs(corr_M[t1]):.4f})")
    print(f"Picco del Valore Mascherato (t2): Campione {t2}  (Correlazione: {abs(corr_V_masked[t2]):.4f})")
    print("-" * 50)
    print(f"IL TUO DELTA (\u03B4) E': {delta} campioni")
    print("="*50 + "\n")

    # ==========================================
    # PLOT PER LA TESI (Questo lo metti nel Capitolo 1)
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # Usiamo abs() per vedere i picchi chiaramente in positivo
    plt.plot(np.abs(corr_M), label='Correlazione Maschera $M$', color='blue', linewidth=1.5)
    plt.plot(np.abs(corr_V_masked), label='Correlazione Valore Mascherato $V \oplus M$', color='red', linewidth=1.5)
    
    # Linee verticali sui picchi
    plt.axvline(t1, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(t2, color='red', linestyle='--', alpha=0.5)
    
    # Annotazione del Delta
    plt.annotate(f'$\Delta$ = {delta}', 
                 xy=((t1+t2)/2, max(abs(corr_M[t1]), abs(corr_V_masked[t2])) * 0.8),
                 ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.title('Identificazione dei Punti di Interesse (POI) - ASCAD Fixed Key')
    plt.xlabel('Campioni Temporali (Time Samples)')
    plt.ylabel('Correlazione di Pearson (Assoluta)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(PATH_PLOTS_ATMEGA, exist_ok=True)
    
    plt.savefig(PATH_PLOTS_ATMEGA + 'leakage_poi_delta.png', dpi=300, bbox_inches='tight')
    print("Grafico salvato come 'leakage_poi_delta.png'. Controlla l'immagine!")
    
if __name__ == '__main__':
    main()
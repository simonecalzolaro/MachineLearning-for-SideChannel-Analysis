import numpy as np
import sys
sys.path.insert(0, '../utils')
from constants import SBOX_ATMEGA, SBOX_RISCURE, PATH_TRACES_ATMEGA, PATH_TRACES_RISCURE_PINATA,MAX_TRACES, BYTE_IDX, DELTA,BTA_BEST_MODELS_PATH_ATMEGA, BTA_BEST_MODELS_PATH_RISCURE_PINATA, BTA_METRICS_PATH_ATMEGA, BTA_METRICS_PATH_RISCURE_PINATA
import feature_selection as fs

from scipy.stats import multivariate_normal
from memory_profiler import memory_usage
import json
from process_traces import process_traces_vectorized
import time
import os
from sklearn.preprocessing import StandardScaler

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SBOX_ATMEGA = SBOX_ATMEGA.flatten()
SBOX_RISCURE = SBOX_RISCURE.flatten()

# Change the Current Working Directory to the project root
os.chdir(PROJECT_ROOT)


def calculate_ge(T_test, ptxs_col, real_key, templates, target, num_exps=100, max_traces=MAX_TRACES, SBOX=SBOX_RISCURE):
    if max_traces is None:
        max_traces = len(T_test)
        
    all_ranks = np.zeros((num_exps, max_traces))
    num_templates = len(templates)
    
    precomputed_logpdf = np.zeros((len(T_test), num_templates))
    for j in range(num_templates):
        precomputed_logpdf[:, j] = templates[j].logpdf(T_test)
    
    for e in range(num_exps):
        idx = np.random.choice(len(T_test), max_traces, replace=False)
        exp_logpdfs = precomputed_logpdf[idx] 
        exp_ptxs = ptxs_col[idx]
        
        key_scores = np.zeros(256)
        for t in range(max_traces):
            pt = exp_ptxs[t]
            for k_guess in range(256):
                sbox_val = SBOX[pt ^ k_guess]
                temp_idx = sbox_val if target == "SBOX_OUT" else fs.HW_TABLE[sbox_val]
                
                key_scores[k_guess] += exp_logpdfs[t, temp_idx]
            
            sorted_candidates = np.argsort(key_scores)[::-1]
            rank = np.where(sorted_candidates == real_key)[0][0]
            all_ranks[e, t] = rank

    return np.mean(all_ranks, axis=0)


def profiling(T, ptxs, key_matrix, byte_target, target, dim, SBOX):
    actual_dim = T.shape[1] 
    N = len(T)
    
    if len(key_matrix.shape) > 1:
        target_keys = key_matrix[:, byte_target]
    else:
        target_keys = key_matrix[byte_target]
        
    sbox_indices = ptxs[:, byte_target] ^ target_keys
    
    if target == "SBOX_OUT":
        labels = np.take(SBOX, sbox_indices)
        num_classes = 256
    elif target == "HW_SO":
        labels = np.take(fs.HW_TABLE, np.take(SBOX, sbox_indices))
        num_classes = 9
    else:
        raise ValueError("Target must be SBOX_OUT or HW_SO")

    templates = []
    
    for val in range(num_classes):
        group_traces = T[labels == val]
        
        if len(group_traces) < 2:
            templates.append({'mean': np.zeros(actual_dim), 'cov': np.eye(actual_dim)})
            continue

        mu = np.mean(group_traces, axis=0)
        sigma = np.cov(group_traces, rowvar=False) + np.eye(actual_dim) * 1e-4 
        
        templates.append({'mean': mu, 'cov': sigma})

    pdf_templates = []
    for temp in templates:
        pdf = multivariate_normal(mean=temp['mean'], cov=temp['cov'], allow_singular=True)
        pdf_templates.append(pdf)
        
    return pdf_templates


def storage_size(feature_choice, pca, lda, templates):
    total_size = 0
    if 'pca' in feature_choice and pca is not None:
        total_size += pca.mean_.nbytes 
        total_size += pca.components_.nbytes
        
    if 'lda' in feature_choice and lda is not None:
        total_size += lda.xbar_.nbytes     
        total_size += lda.scalings_.nbytes 
    
    if templates is not None:
        for temp in templates:
            total_size += temp.mean.nbytes 
            total_size += temp.cov.nbytes
            
    return total_size


def training(feature_choice, X_train, X_test, ptxs_train, key_matrix, target_byte, params, SBOX):
    pca = None
    lda = None
    
    if feature_choice == 'PCA_':
        n_comp = params.get('pca_components')
        if n_comp is None: n_comp = 10
        T, test_T, pca = fs.PCA(X_train, X_test, n_comp)
        templates = profiling(T, ptxs_train, key_matrix, target_byte, "SBOX_OUT", n_comp, SBOX)
        return T, test_T, templates, pca, lda
    
    elif feature_choice == 'LDA_':
        n_comp = params.get('lda_components')
        if n_comp is None: n_comp = 8
        T, test_T, lda = fs.LDA(X_train, X_test, ptxs_train, key_matrix, target_byte, n_comp, SBOX)
        templates = profiling(T, ptxs_train, key_matrix, target_byte, "SBOX_OUT", n_comp, SBOX)
        return T, test_T, templates, pca, lda
    
    elif feature_choice == 'PCA_LDA_':
        n_pca = params.get('pca_components')
        n_lda = params.get('lda_components')
        if n_pca is None: n_pca = 10
        if n_lda is None: n_lda = 8
        T_pca, test_T_pca, pca = fs.PCA(X_train, X_test, n_pca)
        T_lda, test_T_lda, lda = fs.LDA(T_pca, test_T_pca, ptxs_train, key_matrix, target_byte, n_lda, SBOX)
        templates = profiling(T_lda, ptxs_train, key_matrix, target_byte, "SBOX_OUT", n_lda, SBOX)
        return T_lda, test_T_lda, templates, pca, lda    
    
    elif feature_choice == 'PCA_HW_':
        n_comp = params.get('pca_components')
        if n_comp is None: n_comp = 10
        T, test_T, pca = fs.PCA_HW(X_train, X_test, n_comp)
        templates = profiling(T, ptxs_train, key_matrix, target_byte, "HW_SO", n_comp, SBOX)
        return T, test_T, templates, pca, lda
    
    elif feature_choice == 'LDA_HW_': 
        n_comp = params.get('lda_components')
        if n_comp is None: n_comp = 8
        T, test_T, lda = fs.LDA_HW(X_train, X_test, ptxs_train, key_matrix, target_byte, n_comp, SBOX)
        templates = profiling(T, ptxs_train, key_matrix, target_byte, "HW_SO", n_comp, SBOX)
        return T, test_T, templates, pca, lda
    
    elif feature_choice == 'PCA_LDA_HW_':
        n_pca = params.get('pca_components')
        n_lda = params.get('lda_components')
        if n_pca is None: n_pca = 10
        if n_lda is None: n_lda = 8
        T_pca, test_T_pca, pca = fs.PCA_HW(X_train, X_test, n_pca)
        T_lda, test_T_lda, lda = fs.LDA_HW(T_pca, test_T_pca, ptxs_train, key_matrix, target_byte, n_lda, SBOX)
        templates = profiling(T_lda, ptxs_train, key_matrix, target_byte, "HW_SO", n_lda, SBOX)
        return T_lda, test_T_lda, templates, pca, lda
    

def attack(feature_choice, T_test, ptxs_test, test_key, templates, target_byte, SBOX):
    recovered_byte = None
    scoreboard = np.zeros(256)
    num_templates = len(templates)
    
    precomputed_logpdf = np.zeros((len(T_test), num_templates))
    for j in range(num_templates):
        precomputed_logpdf[:, j] = templates[j].logpdf(T_test)
        
    if feature_choice in ['PCA_','LDA_','PCA_LDA_']:
        for i in range(len(T_test)):
            pt = ptxs_test[i]
            for k_guess in range(256):
                sbox_val = SBOX[pt[target_byte] ^ k_guess]
                scoreboard[k_guess] += precomputed_logpdf[i, sbox_val]
        recovered_byte = np.argmax(scoreboard)
    
    elif feature_choice in ['PCA_HW_','LDA_HW_','PCA_LDA_HW_']:
        for i in range(len(T_test)):
            pt = ptxs_test[i]
            for k_guess in range(256):
                sbox_val = SBOX[pt[target_byte] ^ k_guess]
                hw_val = fs.HW_TABLE[sbox_val]
                scoreboard[k_guess] += precomputed_logpdf[i, hw_val]
        recovered_byte = np.argmax(scoreboard)
    
    return recovered_byte


def atmega_metrics():
    base = BTA_BEST_MODELS_PATH_ATMEGA + 'best_'
    #feat_choice = ['PCA_','LDA_','PCA_LDA_','PCA_HW_','LDA_HW_','PCA_LDA_HW_']
    feat_choice = ['PCA_LDA_','PCA_HW_','LDA_HW_','PCA_LDA_HW_']

    operations = ['SUM']

    dataset = PATH_TRACES_ATMEGA
    TRXS = np.load(dataset)['X_train']
    ptxs = np.load(dataset)['ptxs_train']
    KEY = np.load(dataset)['keys_train']
    
    X_TEST = np.load(dataset)['X_test']
    test_ptxs = np.load(dataset)['ptxs_test']
    TEST_KEY = np.load(dataset)['keys_test']
    test_key = TEST_KEY[0]

    sc = StandardScaler()

    target_byte = BYTE_IDX
    N_EXP = np.arange(1, 6, 1) 
    
    for f_c in feat_choice:
        for d_c in operations:
            print(f"\n--- Analyzing {f_c} on {d_c} ---")

            with open(base + f_c[:-1] + '.json', 'r') as f:
                model = json.load(f)
            
            delta = model["delta"]
            target_leakage = "HW_SO" if "HW" in f_c else "SBOX_OUT"

            X_train, X_test = process_traces_vectorized(TRXS, X_TEST, delta, d_c)

            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            metrics = {}

            print("Calculating Training Time...")
            _, _, _, _, _ = training(f_c, X_train, X_test, ptxs, KEY, target_byte, model,SBOX_ATMEGA)
            _, _, _, _, _ = training(f_c, X_train, X_test, ptxs, KEY, target_byte, model,SBOX_ATMEGA)

            durations = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                start = time.perf_counter()
                T, test_T, templates, pca, lda = training(f_c, X_train, X_test, ptxs, KEY, target_byte, model,SBOX_ATMEGA)
                durations.append(time.perf_counter() - start)
            metrics["Training_time"] = np.mean(durations)

            print("Calculating Storage Size...")
            metrics["Storage_size"] = storage_size(f_c, pca, lda, templates)

            print("Calculating Training RAM Usage...")
            peaks = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                # 1. Intervallo molto più aggressivo per funzioni lampo
                mem_usage = memory_usage(
                    (training, (f_c, X_train, X_test, ptxs, KEY, target_byte, model, SBOX_ATMEGA)), 
                    interval=0.001,  # 1 millisecondo invece di 10
                    include_children=True, # Cattura anche processi figli se ci sono
                    multiprocess=True
                )
                # 2. Calcolo del picco relativo all'interno dello stesso array di campionamento
                # Questo evita i problemi di disallineamento della baseline!
                if len(mem_usage) > 1:
                    peak_diff = max(mem_usage) - min(mem_usage)
                else:
                    peak_diff = 0.0
                
                # 3. Se la funzione è letteralmente istantanea e non muove RAM, 
                # registriamo la dimensione empirica degli array in input come proxy o un floor
                if peak_diff <= 0.0:
                    peak_diff = 0.001 # Valore nominale piccolissimo per evitare crash o 0 assoluti nel radar
                peaks.append(peak_diff)
                
            metrics["Training_RAM_usage"] = np.mean(peaks)

            print("Calculating Attack Time...")
            _ = attack(f_c, test_T, test_ptxs, test_key, templates, target_byte,SBOX_ATMEGA)
            _ = attack(f_c, test_T, test_ptxs, test_key, templates, target_byte,SBOX_ATMEGA)
            
            durations = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                start = time.perf_counter()
                _ = attack(f_c, test_T, test_ptxs, test_key, templates, target_byte,SBOX_ATMEGA)
                durations.append(time.perf_counter() - start)
            metrics["Attack_time"] = np.mean(durations)

            print("Calculating Attack RAM Usage...")
            peaks = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                mem_usage = memory_usage(
                    (attack, (f_c, test_T, test_ptxs, test_key, templates, target_byte, SBOX_ATMEGA)), 
                    interval=0.001, 
                    include_children=True,
                    multiprocess=True
                )
                if len(mem_usage) > 1:
                    peak_diff = max(mem_usage) - min(mem_usage)
                else:
                    peak_diff = 0.0
                
                if peak_diff <= 0.0:
                    peak_diff = 0.001
                peaks.append(peak_diff)
                
            metrics["Attack_RAM_usage"] = np.mean(peaks)

            print(f"Calculating GE Curve up to {len(test_T)} traces...")
            ge_curve = calculate_ge(test_T, np.array(test_ptxs)[:, target_byte], test_key[target_byte], templates, target_leakage, 100, len(test_T), SBOX_ATMEGA)
            metrics["GE_curve"] = ge_curve.tolist()
            
            try:
                metrics["GE_crossing_point"] = int(np.where(ge_curve <= 0.5)[0][0] + 1)
            except IndexError:
                metrics["GE_crossing_point"] = len(ge_curve)

            print(f"Saving Metrics for {f_c} and {d_c}...")
            os.makedirs(BTA_METRICS_PATH_ATMEGA, exist_ok=True)
            save_path = os.path.join(BTA_METRICS_PATH_ATMEGA, f'{f_c}{d_c}_metrics.json')
            with open(save_path, 'w') as f:
                json.dump(metrics, f)
            print(f"Done! Metrics saved to {save_path}.")

def riscure_pinata_metrics():
    base = BTA_BEST_MODELS_PATH_RISCURE_PINATA + 'best_'
    #feat_choice = ['PCA_','LDA_','PCA_LDA_','PCA_HW_','LDA_HW_','PCA_LDA_HW_']
    feat_choice = ['LDA_','PCA_LDA_','PCA_HW_','LDA_HW_','PCA_LDA_HW_']

    operations = ['SUM']

    dataset = PATH_TRACES_RISCURE_PINATA
    data_load = np.load(dataset)
    TRXS = data_load['X_train']
    ptxs = data_load['ptxs_train']
    KEY = data_load['keys_train']
    
    X_TEST_FULL = data_load['X_test']
    test_ptxs_full = data_load['ptxs_test']
    TEST_KEY_FULL = data_load['keys_test']

    target_byte = BYTE_IDX
    
    # --- FILTRAGGIO CHIAVE FISSA PER IL TEST SET ---
    print("[*] Estrazione di un subset con chiave fissa dal Test Set...")
    test_keys_byte = TEST_KEY_FULL[:, target_byte]
    most_common_key_byte = np.bincount(test_keys_byte).argmax()
    valid_indices = np.where(test_keys_byte == most_common_key_byte)[0]
    
    print(f"[*] Trovate {len(valid_indices)} tracce con la chiave 0x{most_common_key_byte:02x}.")
    
    X_TEST = X_TEST_FULL[valid_indices]
    test_ptxs = test_ptxs_full[valid_indices]
    test_key = TEST_KEY_FULL[valid_indices[0]] # Chiave rappresentativa fissa
    
    local_max_traces = len(X_TEST)
    print(f"[*] Eseguiremo la GE fino a un massimo di {local_max_traces} tracce.")
    # -----------------------------------------------

    sc = StandardScaler()
    N_EXP = np.arange(1, 6, 1) 
    
    for f_c in feat_choice:
        for d_c in operations:
            print(f"\n--- Analyzing {f_c} on {d_c} ---")

            with open(base + f_c[:-1] + '.json', 'r') as f:
                model = json.load(f)
            
            delta = model["delta"]
            target_leakage = "HW_SO" if "HW" in f_c else "SBOX_OUT"

            X_train, X_test = process_traces_vectorized(TRXS, X_TEST, delta, d_c)

            # Riscure delta is 0 no need to scale for BTA usually, but if you want consistency:
            # X_train = sc.fit_transform(X_train)
            # X_test = sc.transform(X_test)
            
            metrics = {}

            print("Calculating Training Time...")
            _, _, _, _, _ = training(f_c, X_train, X_test, ptxs, KEY, target_byte, model,SBOX_RISCURE)
            _, _, _, _, _ = training(f_c, X_train, X_test, ptxs, KEY, target_byte, model,SBOX_RISCURE)

            durations = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                start = time.perf_counter()
                T, test_T, templates, pca, lda = training(f_c, X_train, X_test, ptxs, KEY, target_byte, model,SBOX_RISCURE)
                durations.append(time.perf_counter() - start)
            metrics["Training_time"] = np.mean(durations)

            print("Calculating Storage Size...")
            metrics["Storage_size"] = storage_size(f_c, pca, lda, templates)

            print("Calculating Training RAM Usage...")
            peaks = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                baseline = memory_usage(-1, max_usage=True) 
                # FIX: Tutti i parametri dentro un'unica tupla!
                mem_usage = memory_usage((training, (f_c, X_train, X_test, ptxs, KEY, target_byte, model, SBOX_RISCURE)), interval=0.01) 
                peaks.append(max(mem_usage) - baseline)
            metrics["Training_RAM_usage"] = np.mean(peaks)

            print("Calculating Attack Time...")
            _ = attack(f_c, test_T, test_ptxs, test_key, templates, target_byte,SBOX_RISCURE)
            _ = attack(f_c, test_T, test_ptxs, test_key, templates, target_byte,SBOX_RISCURE)
            
            durations = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                start = time.perf_counter()
                _ = attack(f_c, test_T, test_ptxs, test_key, templates, target_byte,SBOX_RISCURE)
                durations.append(time.perf_counter() - start)
            metrics["Attack_time"] = np.mean(durations)

            print("Calculating Attack RAM Usage...")
            peaks = []
            for i in N_EXP:
                print(f"Run {i}/{len(N_EXP)}...")
                baseline = memory_usage(-1, max_usage=True) 
                # FIX: Tutti i parametri dentro un'unica tupla!
                mem_usage = memory_usage((attack, (f_c, test_T, test_ptxs, test_key, templates, target_byte, SBOX_RISCURE)), interval=0.01) 
                peaks.append(max(mem_usage) - baseline)
            metrics["Attack_RAM_usage"] = np.mean(peaks)

            print(f"Calculating GE Curve up to {len(test_T)} traces...")
            ge_curve = calculate_ge(test_T, np.array(test_ptxs)[:, target_byte], test_key[target_byte], templates, target_leakage, 100, len(test_T), SBOX_RISCURE)
            metrics["GE_curve"] = ge_curve.tolist()
            
            try:
                metrics["GE_crossing_point"] = int(np.where(ge_curve <= 0.5)[0][0] + 1)
            except IndexError:
                metrics["GE_crossing_point"] = len(ge_curve)

            print(f"Saving Metrics for {f_c} and {d_c}...")
            os.makedirs(BTA_METRICS_PATH_RISCURE_PINATA, exist_ok=True)
            save_path = os.path.join(BTA_METRICS_PATH_RISCURE_PINATA, f'{f_c}{d_c}_metrics.json')
            with open(save_path, 'w') as f:
                json.dump(metrics, f)
            print(f"Done! Metrics saved to {save_path}.")


if __name__ == "__main__":
    dev = sys.argv[1]
    if dev == "atmega":
        print("[*] Avvio della generazione delle metriche per ATMEGA...")
        atmega_metrics()
    elif dev == "riscure_pinata":
        print("[*] Avvio della generazione delle metriche per RISCURE...")
        riscure_pinata_metrics()
    else:
        print("Uso: python3 metric_generator.py [atmega|riscure_pinata]")
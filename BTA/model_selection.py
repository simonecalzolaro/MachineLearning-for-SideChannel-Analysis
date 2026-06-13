import numpy as np
import sys
sys.path.insert(0, '../utils')
sys.path.insert(0,"../traces")
from constants import SBOX_ATMEGA, SBOX_RISCURE, DELTA, BYTE_IDX, PATH_TRACES_ATMEGA, PATH_TRACES_RISCURE_PINATA, MAX_TRACES, BTA_BEST_MODELS_PATH_RISCURE_PINATA, BTA_BEST_MODELS_PATH_ATMEGA
import feature_selection as fs
import json
from process_traces import process_traces_vectorized
from sklearn.preprocessing import StandardScaler 
import torch
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change the Current Working Directory to the project root
os.chdir(PROJECT_ROOT)

# Appiattimento delle SBOX
SBOX_RISCURE = SBOX_RISCURE.flatten()
SBOX_ATMEGA = SBOX_ATMEGA.flatten()

if not torch.cuda.is_available():
    raise RuntimeError("🚨 ERRORE FATALE: Nessuna GPU NVIDIA rilevata!")

device = torch.device("cuda")
print(f"[*] ✅ Utilizzando il device: {torch.cuda.get_device_name(0)}")

@torch.no_grad()
def profiling_gpu(T, ptxs, keys, byte_target, target, sbox):
    actual_dim = T.shape[1]
    N = len(T)
    
    target_ptxs = ptxs[:, byte_target]
    target_keys = keys[:, byte_target]
    sbox_indices = target_ptxs ^ target_keys
    
    labels = np.take(sbox, sbox_indices)
    
    if target == "HW_SO":
        labels = np.take(fs.HW_TABLE, labels)
        num_classes = 9
    else:
        num_classes = 256

    T_tensor = torch.tensor(T, dtype=torch.float64, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    means = []
    covs = []
    
    for val in range(num_classes):
        group = T_tensor[labels_tensor == val]
        
        if len(group) <= actual_dim:
            means.append(torch.zeros(actual_dim, dtype=torch.float64, device=device))
            covs.append(torch.eye(actual_dim, dtype=torch.float64, device=device))
            continue
            
        mu = torch.mean(group, dim=0)
        group_centered = group - mu
        sigma = (group_centered.T @ group_centered) / (len(group) - 1)
        
        sigma += torch.eye(actual_dim, dtype=torch.float64, device=device) * 1e-5
        
        means.append(mu)
        covs.append(sigma)

    all_means = torch.stack(means)
    all_covs = torch.stack(covs)
    
    dist = torch.distributions.MultivariateNormal(all_means, all_covs)
    return dist

@torch.no_grad()
def calculate_ge_gpu(T_test, ptxs_col, real_key_byte, templates_dist, target, num_exps=100, max_traces=MAX_TRACES, sbox=SBOX_RISCURE):
    all_ranks = np.zeros((num_exps, max_traces))
    T_test_tensor = torch.tensor(T_test, dtype=torch.float64, device=device)
    
    pt_key_matrix = np.zeros((256, 256), dtype=int)
    for pt_val in range(256):
        for k_guess in range(256):
            sbox_val = sbox[pt_val ^ k_guess]
            pt_key_matrix[pt_val, k_guess] = sbox_val if target == "SBOX_OUT" else fs.HW_TABLE[sbox_val]
            
    pt_key_tensor = torch.tensor(pt_key_matrix, dtype=torch.long, device=device)

    for e in range(num_exps):
        idx = np.random.choice(len(T_test), max_traces, replace=False)
        exp_traces = T_test_tensor[idx] 
        exp_ptxs = torch.tensor(ptxs_col[idx], dtype=torch.long, device=device) 
        
        log_probs = templates_dist.log_prob(exp_traces.unsqueeze(1)) 
        trace_classes = pt_key_tensor[exp_ptxs] 
        gathered_log_probs = torch.gather(log_probs, 1, trace_classes) 
        
        cumulative_scores = torch.cumsum(gathered_log_probs, dim=0) 
        sorted_candidates = torch.argsort(cumulative_scores, dim=1, descending=True) 
        
        ranks = (sorted_candidates == real_key_byte).nonzero(as_tuple=True)[1] 
        all_ranks[e, :] = ranks.cpu().numpy()

    return np.mean(all_ranks, axis=0)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ==========================================
# ATMEGA SELECTION
# ==========================================
def atmega_selection():
    data_load = np.load(PATH_TRACES_ATMEGA)
    TRXS = data_load['X_train']
    ptxs = data_load['ptxs_train']
    KEY = data_load['keys_train']
    
    X_TEST = data_load['X_test']
    test_ptxs = data_load['ptxs_test']
    TEST_KEY = data_load['keys_test']
    test_key_byte = TEST_KEY[0][BYTE_IDX] 

    feat_red = ["PCA", "LDA", "PCA_LDA", "PCA_HW", "LDA_HW", "PCA_LDA_HW"]
    delta = DELTA
    ops = ["SUM"]
    sc = StandardScaler()
    target_byte = BYTE_IDX

    for t in feat_red:
        for op in ops:
            best = {"Model": t, "Traces_to_GE0": MAX_TRACES, "pca_components": None, "lda_components": None, "operation": op, "delta":delta}
            
            print(f"\n[*] Analisi {t} - Op: {op} - {DELTA} samples shift...")
            X_train, X_test = process_traces_vectorized(TRXS, X_TEST, DELTA, op)
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            if t == "PCA":
                EXP = np.arange(10, 100, 5)
                T_full, test_T_full, _ = fs.PCA(X_train, X_test, max(EXP))
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "SBOX_OUT", SBOX_ATMEGA)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "SBOX_OUT", sbox=SBOX_ATMEGA)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else MAX_TRACES
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_comp,  "delta": delta, "operation": op})

            elif t == "LDA":
                EXP = np.arange(1, 15, 1)
                T_full, test_T_full, _ = fs.LDA(X_train, X_test, ptxs, KEY, target_byte, max(EXP),SBOX_ATMEGA)
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "SBOX_OUT", SBOX_ATMEGA)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "SBOX_OUT", sbox=SBOX_ATMEGA)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else MAX_TRACES
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "lda_components": n_comp, "delta": delta, "operation": op})

            elif t == "PCA_LDA":
                EXP_PCA = np.arange(10, 100, 10)
                EXP_LDA = np.arange(1, 15, 2)
                T_pca_f, test_T_pca_f, _ = fs.PCA(X_train, X_test, max(EXP_PCA))
                for n_pca in EXP_PCA:
                    T_lda_f, test_T_lda_f, _ = fs.LDA(T_pca_f[:, :n_pca], test_T_pca_f[:, :n_pca], ptxs, KEY, target_byte, min(max(EXP_LDA), n_pca),SBOX_ATMEGA)
                    for n_lda in EXP_LDA:
                        if n_lda >= n_pca: continue
                        templates = profiling_gpu(T_lda_f[:, :n_lda], ptxs, KEY, target_byte, "SBOX_OUT", SBOX_ATMEGA)
                        ge = calculate_ge_gpu(test_T_lda_f[:, :n_lda], test_ptxs[:, target_byte], test_key_byte, templates, "SBOX_OUT", sbox=SBOX_ATMEGA)
                        cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else MAX_TRACES
                        if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_pca, "lda_components": n_lda, "delta": delta, "operation": op})

            elif t == "PCA_HW":
                EXP = np.arange(10, 100, 10)
                T_full, test_T_full, _ = fs.PCA_HW(X_train, X_test, max(EXP))
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "HW_SO", SBOX_ATMEGA)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "HW_SO", sbox=SBOX_ATMEGA)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else MAX_TRACES
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_comp, "delta": delta, "operation": op})

            elif t == "LDA_HW":
                EXP = np.arange(1, 9, 1)
                T_full, test_T_full, _ = fs.LDA_HW(X_train, X_test, ptxs, KEY, target_byte, max(EXP),SBOX_ATMEGA)
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "HW_SO", SBOX_ATMEGA)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "HW_SO", sbox=SBOX_ATMEGA)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else MAX_TRACES
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "lda_components": n_comp, "delta": delta, "operation": op})

            elif t == "PCA_LDA_HW":
                EXP_PCA = np.arange(10, 100, 10)
                EXP_LDA = np.arange(1, 9, 1)
                T_pca_f, test_T_pca_f, _ = fs.PCA_HW(X_train, X_test, max(EXP_PCA))
                for n_pca in EXP_PCA:
                    T_lda_f, test_T_lda_f, _ = fs.LDA_HW(T_pca_f[:, :n_pca], test_T_pca_f[:, :n_pca], ptxs, KEY, target_byte, max(EXP_LDA),SBOX_ATMEGA)
                    for n_lda in EXP_LDA:
                        templates = profiling_gpu(T_lda_f[:, :n_lda], ptxs, KEY, target_byte, "HW_SO", SBOX_ATMEGA)
                        ge = calculate_ge_gpu(test_T_lda_f[:, :n_lda], test_ptxs[:, target_byte], test_key_byte, templates, "HW_SO", sbox=SBOX_ATMEGA)
                        cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else MAX_TRACES
                        if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_pca, "lda_components": n_lda, "delta": delta, "operation": op})

        os.makedirs(BTA_BEST_MODELS_PATH_ATMEGA, exist_ok=True)
        nome_file = f"{BTA_BEST_MODELS_PATH_ATMEGA}best_{t}.json"
        with open(nome_file, 'w') as f:
            json.dump(best, f, indent=4, cls=NpEncoder)
                
        print(f"✅ Finito {t}. Best: {best['Traces_to_GE0']} tr.")

# ==========================================
# RISCURE SELECTION
# ==========================================
def riscure_selection():
    data_load = np.load(PATH_TRACES_RISCURE_PINATA)
    TRXS = data_load['X_train']
    ptxs = data_load['ptxs_train']
    KEY = data_load['keys_train']
    
    X_TEST_FULL = data_load['X_test']
    test_ptxs_full = data_load['ptxs_test']
    TEST_KEY_FULL = data_load['keys_test']
    
    target_byte = BYTE_IDX

    print("[*] Estrazione di un subset con chiave fissa dal Test Set...")
    test_keys_byte = TEST_KEY_FULL[:, target_byte]
    
    most_common_key_byte = np.bincount(test_keys_byte).argmax()
    valid_indices = np.where(test_keys_byte == most_common_key_byte)[0]
    
    print(f"[*] Trovate {len(valid_indices)} tracce con la chiave 0x{most_common_key_byte:02x}.")
    
    X_TEST = X_TEST_FULL[valid_indices]
    test_ptxs = test_ptxs_full[valid_indices]
    test_key_byte = most_common_key_byte

    local_max_traces = min(MAX_TRACES, len(X_TEST))
    print(f"[*] Eseguiremo la GE fino a un massimo di {local_max_traces} tracce.")

    feat_red = ["PCA", "LDA", "PCA_LDA", "PCA_HW", "LDA_HW", "PCA_LDA_HW"]
    delta  = 0
    ops = ["SUM"]

    for t in feat_red:
        for op in ops:
            best = {"Model": t, "Traces_to_GE0": local_max_traces, "pca_components": None, "lda_components": None, "operation": op, "delta":delta}
            
            print(f"\n[*] Analisi {t} - Op: {op}...")

            X_train, X_test = process_traces_vectorized(TRXS, X_TEST, delta, op)

            if t == "PCA":
                EXP = np.arange(10, 100, 5)
                T_full, test_T_full, _ = fs.PCA(X_train, X_test, max(EXP))
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "SBOX_OUT", SBOX_RISCURE)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "SBOX_OUT", max_traces=local_max_traces, sbox=SBOX_RISCURE)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else local_max_traces
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_comp,  "delta": delta, "operation": op})

            elif t == "LDA":
                EXP = np.arange(1, 15, 1)
                T_full, test_T_full, _ = fs.LDA(X_train, X_test, ptxs, KEY, target_byte, max(EXP),SBOX_RISCURE)
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "SBOX_OUT", SBOX_RISCURE)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "SBOX_OUT", max_traces=local_max_traces, sbox=SBOX_RISCURE)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else local_max_traces
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "lda_components": n_comp, "delta": delta, "operation": op})

            elif t == "PCA_LDA":
                EXP_PCA = np.arange(10, 100, 10)
                EXP_LDA = np.arange(1, 15, 2)
                T_pca_f, test_T_pca_f, _ = fs.PCA(X_train, X_test, max(EXP_PCA))
                for n_pca in EXP_PCA:
                    T_lda_f, test_T_lda_f, _ = fs.LDA(T_pca_f[:, :n_pca], test_T_pca_f[:, :n_pca], ptxs, KEY, target_byte, min(max(EXP_LDA), n_pca),SBOX_RISCURE)
                    for n_lda in EXP_LDA:
                        if n_lda >= n_pca: continue
                        templates = profiling_gpu(T_lda_f[:, :n_lda], ptxs, KEY, target_byte, "SBOX_OUT", SBOX_RISCURE)
                        ge = calculate_ge_gpu(test_T_lda_f[:, :n_lda], test_ptxs[:, target_byte], test_key_byte, templates, "SBOX_OUT", max_traces=local_max_traces, sbox=SBOX_RISCURE)
                        cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else local_max_traces
                        if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_pca, "lda_components": n_lda, "delta": delta, "operation": op})

            elif t == "PCA_HW":
                EXP = np.arange(10, 100, 10)
                T_full, test_T_full, _ = fs.PCA_HW(X_train, X_test, max(EXP))
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "HW_SO", SBOX_RISCURE)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "HW_SO", max_traces=local_max_traces, sbox=SBOX_RISCURE)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else local_max_traces
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_comp, "delta": delta, "operation": op})

            elif t == "LDA_HW":
                EXP = np.arange(1, 9, 1)
                T_full, test_T_full, _ = fs.LDA_HW(X_train, X_test, ptxs, KEY, target_byte, max(EXP), SBOX_RISCURE)
                for n_comp in EXP:
                    templates = profiling_gpu(T_full[:, :n_comp], ptxs, KEY, target_byte, "HW_SO", SBOX_RISCURE)
                    ge = calculate_ge_gpu(test_T_full[:, :n_comp], test_ptxs[:, target_byte], test_key_byte, templates, "HW_SO", max_traces=local_max_traces, sbox=SBOX_RISCURE)
                    cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else local_max_traces
                    if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "lda_components": n_comp, "delta": delta, "operation": op})

            elif t == "PCA_LDA_HW":
                EXP_PCA = np.arange(10, 100, 10)
                EXP_LDA = np.arange(1, 9, 1)
                T_pca_f, test_T_pca_f, _ = fs.PCA_HW(X_train, X_test, max(EXP_PCA))
                for n_pca in EXP_PCA:
                    T_lda_f, test_T_lda_f, _ = fs.LDA_HW(T_pca_f[:, :n_pca], test_T_pca_f[:, :n_pca], ptxs, KEY, target_byte, max(EXP_LDA), SBOX_RISCURE)
                    for n_lda in EXP_LDA:
                        templates = profiling_gpu(T_lda_f[:, :n_lda], ptxs, KEY, target_byte, "HW_SO", SBOX_RISCURE)
                        ge = calculate_ge_gpu(test_T_lda_f[:, :n_lda], test_ptxs[:, target_byte], test_key_byte, templates, "HW_SO", max_traces=local_max_traces, sbox=SBOX_RISCURE)
                        cp = np.where(ge<0.5)[0][0]+1 if np.any(ge<0.5) else local_max_traces
                        if cp < best["Traces_to_GE0"]: best.update({"Traces_to_GE0": cp, "pca_components": n_pca, "lda_components": n_lda, "delta": delta, "operation": op})

        os.makedirs(BTA_BEST_MODELS_PATH_RISCURE_PINATA, exist_ok=True)
        nome_file = f"{BTA_BEST_MODELS_PATH_RISCURE_PINATA}best_{t}.json"
        with open(nome_file, 'w') as f:
            json.dump(best, f, indent=4, cls=NpEncoder)
                
        print(f"✅ Finito {t}. Best: {best['Traces_to_GE0']} tr.")

if __name__ == "__main__":
    dev = sys.argv[1]
    if dev == "atmega":
        print("[*] Avvio della selezione del modello per ATMEGA...")
        atmega_selection()
    elif dev == "riscure_pinata":
        print("[*] Avvio della selezione del modello per RISCURE...")
        riscure_selection()
    else:
        print("Uso: python3 model_selection.py [atmega|riscure_pinata]")
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from memory_profiler import memory_usage

# Custom Utils
sys.path.insert(0, '../utils')
import constants
import results
from load_data import TestDataLoader
from constants import CNN_MODEL_PATH_RISCURE, MLP_MODEL_PATH_RISCURE, MLP_PRE_MODEL_PATH_RISCURE, PATH_TRACES_RISCURE_PINATA, BYTE_IDX, RISCURE_TEST_SIZE, SBOX_RISCURE

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

def plot_ge(ge, output_path, show=False):
    f, ax = plt.subplots(figsize=(15,8))
    v_value = results.min_att_tr(ge)
    if v_value > len(ge):
        ax.axvline(v_value + 300, color='r', linestyle='--', linewidth=3, label=f'GE = {len(ge)}+')
    else:
        ax.axvline(v_value, color='r', linestyle='--', linewidth=3, label=f'GE = {v_value}')
    
    ax.plot(range(1, len(ge)+1), ge, marker='o', color='b')
    x_label_step = 30
    ax.set_xlim([0, len(ge)])
    ax.set_title('Guessing Entropy')
    ax.set_xticks(range(0, len(ge)+1, x_label_step), labels=range(0, len(ge)+1, x_label_step))
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('GE')
    ax.set_yticks(range(0, 100+1, 10), labels=range(0, 100+1, 10))
    ax.grid(alpha=0.2)
    ax.legend(loc='best')
    f.savefig(output_path, bbox_inches='tight', dpi=600)
    if show: plt.show()
    plt.close(f)

def to_binary_matrix(array):
    return np.unpackbits(np.array(array, dtype=np.uint8).reshape(-1, 1), axis=1)

def bits_to_bytes_probs(preds_bits):
    bits_matrix = np.unpackbits(np.arange(256, dtype=np.uint8).reshape(256, 1), axis=1)
    preds_bytes = np.zeros((preds_bits.shape[0], 256))
    for i in range(256):
        bits = bits_matrix[i]
        p_byte = np.prod(np.where(bits == 1, preds_bits, 1 - preds_bits), axis=1)
        preds_bytes[:, i] = p_byte
    return preds_bytes

def ge(model, x_test, ptx_bytes, true_key_byte, n_exp, target,sbox):
    print(f'TRUEKEYBYTE = {true_key_byte}')
    tr_per_exp = int(x_test.shape[0] / n_exp)
    ranks_per_exp = []

    for i in range(n_exp):
        start = i * tr_per_exp
        stop = start + tr_per_exp
        x_batch = x_test[start:stop]
        pltxt_bytes_batch = ptx_bytes[start:stop]
        curr_preds = model.predict(x_batch, verbose=0)
        
        if curr_preds.shape[1] == 8:
            curr_preds = bits_to_bytes_probs(curr_preds)

        final_rankings = results.compute_final_rankings(curr_preds, pltxt_bytes_batch, target,sbox)
        true_kb_ranks = np.array([kbs.index(true_key_byte) for kbs in final_rankings]) 
        ranks_per_exp.append(true_kb_ranks)

    ranks_per_exp = np.vstack(ranks_per_exp) 
    ge_array = np.mean(ranks_per_exp, axis=0) 
    return ge_array

def main():
    ptx_mode = sys.argv[1].lower()
    dataset = sys.argv[2].lower()
    target = str(sys.argv[3])
    out_mode = sys.argv[4].lower()
    arch = sys.argv[5]
    b = BYTE_IDX

    assert ptx_mode in ['none', 'scalar', 'binary']
    assert dataset in ['random', 'fixed']
    assert target in ['KEY', 'SBOX_OUT', 'HW_SO']
    assert out_mode in ['categorical', 'binary']
    if out_mode == 'binary':
        assert target == 'SBOX_OUT'

    tot_traces = RISCURE_TEST_SIZE

    if arch == "CNN_ZAID":
        MODEL_ROOT = f'{CNN_MODEL_PATH_RISCURE}TRAINING/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
        RESULTS_ROOT = f'{CNN_MODEL_PATH_RISCURE}ATTACK/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
    elif arch == "MLP":
        MODEL_ROOT = f'{MLP_MODEL_PATH_RISCURE}TRAINING/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
        RESULTS_ROOT = f'{MLP_MODEL_PATH_RISCURE}ATTACK/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
    elif arch == "MLP_PRE":
        MODEL_ROOT = f'{MLP_PRE_MODEL_PATH_RISCURE}TRAINING/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
        RESULTS_ROOT = f'{MLP_PRE_MODEL_PATH_RISCURE}ATTACK/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
    
    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    id_train = f'{target}'
    if ptx_mode != 'none': id_train += f'_ptx_{ptx_mode}'
    if out_mode == 'binary': id_train += '_out_binary'

    MODEL_FILENAME = os.path.join(MODEL_ROOT, f'{id_train}_model.h5')
    GE_CSV_FILENAME = os.path.join(RESULTS_ROOT, f'{id_train}_ge.csv')
    GE_PLOT_FILENAME = os.path.join(RESULTS_ROOT, f'{id_train}_ge_plot.png')
    TIME_FILENAME = os.path.join(RESULTS_ROOT, f'{id_train}_attack_time.txt')
    RAM_FILENAME = os.path.join(RESULTS_ROOT, f'{id_train}_attack_ram.txt')

    test_files = [f'{PATH_TRACES_RISCURE_PINATA}']
    delta = 0
    
    test_dl = TestDataLoader(
        test_files,
        tot_traces=tot_traces,
        target=target,
        byte_idx=b,
        out_mode=out_mode,
        delta=delta,
        op="SUM",
        sbox=SBOX_RISCURE.flatten() # <--- Passa S-Box
    )
    x_test, _, pbs_test, tkb_test = test_dl.load()

    if ptx_mode == 'scalar':
        pbs_test_scaled = (pbs_test / 255.0).astype(np.float32).reshape(-1, 1)
        x_test = np.append(x_test, pbs_test_scaled, axis=1)
    elif ptx_mode == 'binary':
        x_test = np.append(x_test, to_binary_matrix(pbs_test).astype(np.float32), axis=1)

    print(f"Loading model from: {MODEL_FILENAME}")
    test_model = load_model(MODEL_FILENAME, safe_mode=False)

    print("Running GE Attack and Profiling Resources...")
    true_key = int(tkb_test[0][0] if isinstance(tkb_test[0], (list, np.ndarray)) else tkb_test[0])

    def run_attack():
        return ge(
            model=test_model,
            x_test=x_test,
            ptx_bytes=pbs_test,
            true_key_byte=true_key,
            n_exp=10,
            target=target,
            sbox=SBOX_RISCURE.flatten()
        )

    baseline = memory_usage(-1, max_usage=True)
    start_time = time.time()
    mem_tuple = memory_usage((run_attack,), interval=0.1, retval=True)
    elapsed_time = time.time() - start_time

    mem_usage_array = mem_tuple[0]
    ge_result = mem_tuple[1]
    peak_ram_mb = max(mem_usage_array) - baseline

    np.savetxt(GE_CSV_FILENAME, ge_result, delimiter=',')
    plot_ge(ge_result[:1000], GE_PLOT_FILENAME)

    return elapsed_time, peak_ram_mb, TIME_FILENAME, RAM_FILENAME

if __name__ == "__main__":
    start = time.time()
    elapsed_time, peak_ram_mb, time_path, ram_path = main()  
    elapsed = time.time() - start 
    print(f'Elapsed execution time: {elapsed:.2f} s')
    print(f'Attack Phase Peak RAM overhead: {peak_ram_mb:.2f} MB')
    with open(time_path, 'w') as f: f.write(f'Elapsed time: {elapsed_time:.2f} s\n')
    with open(ram_path, 'w') as f: f.write(f'Peak RAM overhead: {peak_ram_mb:.2f} MB\n')
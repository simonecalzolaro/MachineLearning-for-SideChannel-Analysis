# Basics
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import os
import joblib

# Custom
sys.path.insert(0, '../utils')
from constants import CNN_MODEL_PATH_RISCURE, MLP_MODEL_PATH_RISCURE, MLP_PRE_MODEL_PATH_RISCURE, PATH_TRACES_RISCURE_PINATA, BYTE_IDX, RISCURE_TRAIN_SIZE, SBOX_RISCURE
from load_data import SplitDataLoader
from network import Network
from memory_profiler import memory_usage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

EPOCHS = 800
VERBOSE = 1

def plot_history(history, metric, output_path, show=False):
    f = plt.figure(figsize=(10,10))
    train_label = f'train_{metric}'
    val_label = f'val_{metric}'
    title = f'Train and Val {metric.title()}' 
    plt.plot(history[metric], label=train_label)
    plt.plot(history[val_label], label=val_label)
    if metric == 'accuracy':
        plt.axhline(y=1/256, color='r', linewidth=3, linestyle='--', label='Random-Guesser Accuracy')
    plt.title(title)
    plt.ylabel(metric.title())
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    f.savefig(f'{output_path}.svg', bbox_inches='tight', dpi=600)
    f.savefig(f'{output_path}.png', bbox_inches='tight', dpi=600)
    if show: plt.show()
    plt.close(f)

def to_binary_matrix(array):
    return np.unpackbits(np.array(array, dtype=np.uint8).reshape(-1, 1), axis=1)

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

    tot_traces = RISCURE_TRAIN_SIZE

    if arch == "CNN_ZAID":
        RES_ROOT = f'{CNN_MODEL_PATH_RISCURE}TRAINING/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
    elif arch == "MLP":
        RES_ROOT = f'{MLP_MODEL_PATH_RISCURE}TRAINING/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
    elif arch == "MLP_PRE":
        RES_ROOT = f'{MLP_PRE_MODEL_PATH_RISCURE}TRAINING/{target}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'

    IMAGES = RES_ROOT + '/plots'
    os.makedirs(RES_ROOT, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)

    id_train = f'{target}'
    if ptx_mode != 'none': id_train += f'_ptx_{ptx_mode}'
    if out_mode == 'binary': id_train += '_out_binary'

    train_files = [f'{PATH_TRACES_RISCURE_PINATA}']
    MODEL_FILENAME = os.path.join(RES_ROOT, f'{id_train}_model.h5')
    LOSS_HISTORY_FILENAME = os.path.join(IMAGES, f'{id_train}_loss_history')
    ACC_HISTORY_FILENAME = os.path.join(IMAGES, f'{id_train}_acc_history')
    SCALER_FILENAME = os.path.join(RES_ROOT, f'{id_train}_scaler.save')

    delta = 0

    train_dl = SplitDataLoader(
        train_files,
        tot_traces=tot_traces,
        train_size=0.9,
        target=target,
        byte_idx=b,
        out_mode=out_mode,
        delta=delta,
        op="SUM",
        sbox=SBOX_RISCURE.flatten() # <--- Passa S-Box
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, ptx_train, _ = train_data
    x_val, y_val, ptx_val, _ = val_data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    joblib.dump(scaler, SCALER_FILENAME)

    if "MLP" in arch:
        model_type = f'MLP_{target}'
    else:
        model_type = f'ZAID_{target}'

    if ptx_mode == 'scalar':
        ptx_train_scaled = (ptx_train / 255.0).astype(np.float32).reshape(-1, 1)
        ptx_val_scaled = (ptx_val / 255.0).astype(np.float32).reshape(-1, 1)
        x_train = np.append(x_train, ptx_train_scaled, axis=1)
        x_val = np.append(x_val, ptx_val_scaled, axis=1)
        model_type += '_ptx_scalar'
    elif ptx_mode == 'binary':
        x_train = np.append(x_train, to_binary_matrix(ptx_train).astype(np.float32), axis=1)
        x_val = np.append(x_val, to_binary_matrix(ptx_val).astype(np.float32), axis=1)
        model_type += '_ptx_binary'

    if out_mode == 'binary':
        model_type += '_out_binary'

    print(f'Model Configuration: {model_type}')
    input_dim = x_train.shape[1]
    print(f'Final Input Dimension: {input_dim}')

    attack_net = Network(model_type, {'batch_size': 50})
    attack_net.build_model(input_dim=input_dim)
    attack_net.add_checkpoint_callback(MODEL_FILENAME)

    print('Training start')
    def train_nn():
        return attack_net.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=attack_net.hp['batch_size'],
            callbacks=attack_net.callbacks,
            verbose=VERBOSE
        ).history

    baseline = memory_usage(-1, max_usage=True)
    start_time = time.time()
    mem_tuple = memory_usage((train_nn,), interval=1.0, retval=True)
    elapsed_time = time.time() - start_time

    mem_usage_array = mem_tuple[0]
    history = mem_tuple[1]
    peak_ram_mb = max(mem_usage_array) - baseline

    plot_history(history, 'loss', LOSS_HISTORY_FILENAME)
    plot_history(history, 'accuracy', ACC_HISTORY_FILENAME)

    time_path = f'{RES_ROOT}/{id_train}_training_time.txt'
    ram_path = f'{RES_ROOT}/{id_train}_training_ram.txt'
    
    return elapsed_time, peak_ram_mb, time_path, ram_path

if __name__ == '__main__':
    start = time.time()
    elapsed_time, peak_ram_mb, time_path, ram_path = main()  
    elapsed = time.time() - start 
    print(f'Elapsed time: {elapsed:.2f} s')
    print(f'Peak RAM overhead: {peak_ram_mb:.2f} MB')
    with open(time_path, 'w') as f: f.write(f'Elapsed time: {elapsed:.2f} s\n')
    with open(ram_path, 'w') as f: f.write(f'Peak RAM overhead: {peak_ram_mb:.2f} MB\n')
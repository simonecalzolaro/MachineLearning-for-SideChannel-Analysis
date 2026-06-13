# Basics
import json
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Custom
import sys
import helpers
from constants import SBOX, PATH_TRACES, PATH_MODELS, PATH_METRICS, PATH_RESULTS,DELTA,BYTE
import visualization as vis
from data_loader import SplitDataLoader # Updated import
from hp_tuner import HPTuner

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # allocate memory over time, so i can see the usage in real time

N_MODELS = 15
N_GEN = 13
EPOCHS = 100
HP = {
    'hidden_layers':  [1, 2, 3, 4, 5, 6, 7],
    'hidden_neurons': [100, 200, 300, 400, 500],
    'dropout_rate':   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'l2':             [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':      ['adam', 'rmsprop', 'sgd'],
    'learning_rate':  [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':     [128, 256, 512, 1024],
}

def to_binary_matrix(array):
    """Helper to convert a 1D array of integers to a 2D matrix of bits"""
    return np.unpackbits(np.array(array, dtype=np.uint8).reshape(-1, 1), axis=1)

def main():

    """
    Performs hyperparameter tuning for an MLP model with the specified settings.
    Settings parameters (provided in order via command line):
        - ptx_mode: 'none', 'scalar', or 'binary'
        - dataset: 'random' or 'fixed'
        - target: 'SBOX_OUT', 'KEY', or 'HW_SO'
        - out_mode: 'categorical' or 'binary'
        - train_devs: Devices to use during training

    The result is a JSON file containing the best hyperparameters.
    """

    ptx_mode = sys.argv[1].lower()
    dataset = sys.argv[2].lower()
    TARGET = str(sys.argv[3])
    out_mode = sys.argv[4].lower()
    #train_devs = sys.argv[5:]
    b = BYTE
    
    assert ptx_mode in ['none', 'scalar', 'binary'], "ptx_mode must be none, scalar, or binary"
    assert dataset in ['random', 'fixed'], "dataset must be random or fixed"
    assert TARGET in ['KEY', 'SBOX_OUT', 'HW_SO'], "Invalid target"
    assert out_mode in ['categorical', 'binary'], "out_mode must be categorical or binary"
    
    if out_mode == 'binary':
        assert TARGET == 'SBOX_OUT', "Binary output encoding is only valid for SBOX_OUT target."

    # tuning done with 50k traces for speed
    tot_traces = 50000

    # Ensure paths match new_training.py structure
    RES_ROOT = f'{PATH_RESULTS}HP_TUNING/{TARGET}/{dataset}_key_{out_mode}_out_{ptx_mode}_ptx'
    IMAGES = RES_ROOT + '/plots'
    # make dir if non existant
    os.makedirs(RES_ROOT, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)

    id_train = f'{TARGET}'
    if ptx_mode != 'none': id_train += f'_ptx_{ptx_mode}'
    if out_mode == 'binary': id_train += '_out_binary'

    train_files = [f'{PATH_TRACES}'] # Update to new path

    LOSS_HIST_FILE = RES_ROOT + f'/{id_train}_hp_loss_hist_data.csv'
    ACC_HIST_FILE = RES_ROOT + f'/{id_train}_hp_acc_hist_data.csv'
    HISTORY_PLOT = IMAGES + f'/{id_train}_hp_tuning_history.svg'
    HP_PATH = f'{RES_ROOT}/{id_train}_hp.json'


    # Get data
    train_dl = SplitDataLoader(
        train_files,
        tot_traces=tot_traces,
        train_size=0.9,
        target=TARGET,
        byte_idx=b,
        out_mode=out_mode, # Pass out_mode to data loader
        delta=DELTA,
        op="DIFF"
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, ptx_train, _ = train_data
    x_val, y_val, ptx_val, _ = val_data

    # scale data in [0-1] range
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    model_type = f'MLP_{TARGET}'
    
    # --- PTX ENCODING LOGIC ---
    # --- PTX ENCODING LOGIC ---
    if ptx_mode == 'scalar':
        # Aggiunto .reshape(-1, 1) per evitare il ValueError su axis=1
        ptx_train_scaled = (ptx_train / 255.0).astype(np.float32).reshape(-1, 1)
        ptx_val_scaled = (ptx_val / 255.0).astype(np.float32).reshape(-1, 1)
        
        x_train = np.append(x_train, ptx_train_scaled, axis=1)
        x_val = np.append(x_val, ptx_val_scaled, axis=1)
        model_type += '_ptx_scalar'
    elif ptx_mode == 'binary':
        # to_binary_matrix gestisce già il reshape internamente
        x_train = np.append(x_train, to_binary_matrix(ptx_train).astype(np.float32), axis=1)
        x_val = np.append(x_val, to_binary_matrix(ptx_val).astype(np.float32), axis=1)
        model_type += '_ptx_binary'

    if out_mode == 'binary':
        model_type += '_out_binary'

    print(f'model: {model_type} (ptx_mode = {ptx_mode}, out_mode = {out_mode})')

    # HP Tuning via Genetic Algorithm
    hp_tuner = HPTuner(
        model_type=model_type,
        hp_space=HP,
        n_models=N_MODELS,
        n_epochs=EPOCHS
    )

    # In /home/scalzolaro/masked/NN/hp_tuning.py

    # ... [tutto il codice prima rimane uguale fino a hp_tuner.genetic_algorithm] ...

    best_hp = hp_tuner.genetic_algorithm(
        n_gen=N_GEN,
        selection_perc=0.3,
        second_chance_prob=0.2,
        mutation_prob=0.2,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )
    
    print(f"\n--- MIGLIORI HP TROVATI ---")
    print(best_hp)

    # 1. SALVATAGGIO JSON IMMEDIATO (Niente più dati persi!)
    try:
        os.makedirs(os.path.dirname(HP_PATH), exist_ok=True)
        with open(HP_PATH, 'w') as jfile:
            json.dump(best_hp, jfile, indent=4)
        print(f"[SUCCESS] Migliori HP salvati in: {HP_PATH}")
    except Exception as e:
        print(f"[CRITICAL ERROR] Fallimento nel salvataggio JSON: {e}")
        # Salvataggio di emergenza
        with open(HP_PATH + ".txt", 'w') as f:
            f.write(str(best_hp))

    # 2. ESTRAZIONE E SALVATAGGIO CSV
    b_history = hp_tuner.best_history
    actual_epochs = len(b_history['loss']) 
    
    # Loss
    loss_data = np.vstack((np.arange(actual_epochs)+1, b_history['loss'], b_history['val_loss'])).T
    helpers.save_csv(data=loss_data, columns=['Epochs', 'Loss', 'Val_Loss'], output_path=LOSS_HIST_FILE)

    # Accuracy
    acc_data = np.vstack((np.arange(actual_epochs)+1, b_history['accuracy'], b_history['val_accuracy'])).T
    helpers.save_csv(data=acc_data, columns=['Epochs', 'Acc', 'Val_Acc'], output_path=ACC_HIST_FILE)

    # 3. VISUALIZZAZIONE SICURA (Se fallisce, pace, i dati sono già salvati)
    try:
        vis.plot_history(b_history, HISTORY_PLOT)
        print(f"[SUCCESS] Grafici generati in: {HISTORY_PLOT}")
    except AttributeError:
        print("[WARNING] La funzione plot_history non esiste in visualization.py. Grafico saltato.")
    except Exception as e:
        print(f"[WARNING] Errore imprevisto nei plot: {e}")

    return f'{RES_ROOT}/{id_train}_hp_time.txt'

# ... [resto del file (if __name__ == '__main__':)] ...


if __name__ == '__main__':
    start = time.time()
    time_path = main()  
    
    elapsed = time.time() - start 
    print(f'Elapsed time: {elapsed:.2f} s')
    
    with open(time_path, 'w') as f:
        f.write(f'Elapsed time: {elapsed:.2f} s\n')
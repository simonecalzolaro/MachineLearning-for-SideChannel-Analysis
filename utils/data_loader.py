import h5py
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import trsfile
from constants import RAW_TRACES_RISCURE_PINATA, RAW_TRACES_ATMEGA, PATH_TRACES_RISCURE_PINATA, PATH_TRACES_ATMEGA
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change the Current Working Directory to the project root
os.chdir(PROJECT_ROOT)


def load_h5(h5_file_path):
    print(f"Loading ASCAD database from {h5_file_path}...")
    
    with h5py.File(h5_file_path, "r") as in_file:
        
        X_train = np.array(in_file['Profiling_traces']['traces'], dtype=np.float32)
        ptxs_train = np.array(in_file['Profiling_traces']['metadata']['plaintext'], dtype=np.uint8)
        keys_train = np.array(in_file['Profiling_traces']['metadata']['key'], dtype=np.uint8)
        
        masks_train = np.array(in_file['Profiling_traces']['metadata']['masks'], dtype=np.uint8)

        X_test = np.array(in_file['Attack_traces']['traces'], dtype=np.float32)
        ptxs_test = np.array(in_file['Attack_traces']['metadata']['plaintext'], dtype=np.uint8)
        keys_test = np.array(in_file['Attack_traces']['metadata']['key'], dtype=np.uint8)
        
        masks_test = np.array(in_file['Attack_traces']['metadata']['masks'], dtype=np.uint8)

    return X_train, ptxs_train, keys_train, masks_train, X_test, ptxs_test, keys_test, masks_test

def load_trs(files_list):
    traces = []
    ptxs = []
    keys = []
    for file in files_list:
        print(f"Loading TRS file: {file}")
        with trsfile.open(file, 'r') as f:
            for trace in f:
                traces.append(trace.samples)
                ptxs.append(trace.get_input())
                keys.append(trace.get_key())
    return np.array(traces,dtype=np.float32), np.array(ptxs,dtype=np.uint8), np.array(keys,dtype=np.uint8)


if __name__ == "__main__":

    
    riscure_traces_train, riscure_ptxs_train, riscure_keys_train = load_trs([RAW_TRACES_RISCURE_PINATA + "D1.trs",RAW_TRACES_RISCURE_PINATA + "D2.trs"])
    riscure_traces_test, riscure_ptxs_test, riscure_keys_test = load_trs([RAW_TRACES_RISCURE_PINATA + "D1_test.trs",RAW_TRACES_RISCURE_PINATA + "D2_test.trs"])

    
    atmega_traces_train, atmega_ptxs_train, atmega_keys_train ,atmega_masks_train,atmega_traces_test, atmega_ptxs_test, atmega_keys_test, atmega_masks_test = load_h5(RAW_TRACES_ATMEGA+"ASCAD_RANDOMKEY.h5")

    print("Data loaded successfully!")

    riscure_traces_train, riscure_ptxs_train, riscure_keys_train = shuffle(riscure_traces_train, riscure_ptxs_train, riscure_keys_train, random_state=42)
    riscure_traces_test, riscure_ptxs_test, riscure_keys_test = shuffle(riscure_traces_test, riscure_ptxs_test, riscure_keys_test, random_state=42)
    atmega_traces_train, atmega_ptxs_train, atmega_keys_train, atmega_masks_train = shuffle(atmega_traces_train, atmega_ptxs_train, atmega_keys_train, atmega_masks_train, random_state=42)
    atmega_traces_test, atmega_ptxs_test, atmega_keys_test, atmega_masks_test = shuffle(atmega_traces_test, atmega_ptxs_test, atmega_keys_test, atmega_masks_test, random_state=42)
    print("Data shuffled successfully!")
    
    # Normalization
    sc = StandardScaler()
    #Riscure
    riscure_traces_train = sc.fit_transform(riscure_traces_train)
    riscure_traces_test = sc.transform(riscure_traces_test)

    #Atmega
    atmega_traces_train = sc.fit_transform(atmega_traces_train)
    atmega_traces_test = sc.transform(atmega_traces_test)

    print("Data normalized successfully!")

    print("Shapes:")
    print(f"Riscure train: {riscure_traces_train.shape}")
    print(f"Riscure test: {riscure_traces_test.shape}")
    print(f"Atmega train: {atmega_traces_train.shape}")
    print(f"Atmega test: {atmega_traces_test.shape}")

    np.savez_compressed(PATH_TRACES_RISCURE_PINATA, 
                        X_train=riscure_traces_train, ptxs_train=riscure_ptxs_train, keys_train=riscure_keys_train,
                        X_test=riscure_traces_test, ptxs_test=riscure_ptxs_test, keys_test=riscure_keys_test)
    
    np.savez_compressed(PATH_TRACES_ATMEGA, 
                        X_train=atmega_traces_train, ptxs_train=atmega_ptxs_train, keys_train=atmega_keys_train, masks_train=atmega_masks_train,
                        X_test=atmega_traces_test, ptxs_test=atmega_ptxs_test, keys_test=atmega_keys_test, masks_test=atmega_masks_test)
    
    print("Data saved successfully!")
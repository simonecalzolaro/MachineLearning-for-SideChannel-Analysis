# Basics
import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from abc import ABC, abstractmethod

# Custom
import aes
import sys
sys.path.insert(0, '../utils')
import constants
# CAMBIA l'import se hai chiamato il file in modo diverso!
from process_traces_NN import process_traces_vectorized 
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

class DataLoader(ABC):
    def __init__(self, trace_files, tot_traces, target, byte_idx=None, out_mode='categorical', delta=0, op="DIFF", sbox=None):
        self.trace_files = trace_files
        self.n_tr_per_config = int(tot_traces / len(trace_files))
        self.target = target
        self.n_classes = constants.N_CLASSES[target]
        self.byte_idx = byte_idx
        self.out_mode = out_mode 
        self.delta = delta       
        self.operation = op      
        self.sbox = sbox # <--- Inizializzazione S-Box dinamica
        
    @staticmethod
    def _shuffle(x, y, pbs, tkbs):
        to_shuffle = list(zip(x, y, pbs, tkbs))
        random.shuffle(to_shuffle)
        x, y, pbs, tkbs = zip(*to_shuffle)

        x = np.vstack(x)
        y = np.vstack(y)
        pbs = np.vstack(pbs)   
        tkbs = np.vstack(tkbs) 

        return x, y, pbs, tkbs

    @abstractmethod
    def load(self):
        pass

class SplitDataLoader(DataLoader):
    def __init__(self, trace_files, tot_traces, train_size, target, byte_idx=None, out_mode='categorical', delta=0, op="SUM", sbox=None):
        super().__init__(trace_files, tot_traces, target, byte_idx, out_mode, delta, op, sbox)
        self.n_train_tr_per_config = int(train_size * self.n_tr_per_config)

    def load(self):
        x_train, y_train, pbs_train, tkbs_train = [], [], [], []
        x_val, y_val, pbs_val, tkbs_val = [], [], [], []

        for tfile in self.trace_files:
            data = np.load(tfile)
            TRXS = data['X_train']
            ptxs = data['ptxs_train']
            KEYS = data['keys_train']

            TRXS = process_traces_vectorized(TRXS, self.delta, self.operation)

            X_sub = TRXS[:self.n_tr_per_config]
            p_sub = ptxs[:self.n_tr_per_config]
            k_sub = KEYS[:self.n_tr_per_config] 

            X_sub, p_sub, k_sub = shuffle(X_sub, p_sub, k_sub, random_state=42)

            config_s, config_l, config_p, config_k = [], [], [], []

            for i, x in enumerate(X_sub):
                pt = p_sub[i]
                current_key = k_sub[i] 
                l = aes.labels_from_key(pt, current_key, self.target, sbox=self.sbox) # <--- Uso S-Box dinamica

                if self.byte_idx is not None:
                    l = l[self.byte_idx]
                    pt = pt[self.byte_idx]
                    k_val = current_key[self.byte_idx]
                else:
                    k_val = current_key

                if self.out_mode == 'categorical':
                    l = to_categorical(l, self.n_classes)

                config_s.append(x)
                config_p.append(pt)
                config_k.append(k_val)
                config_l.append(l)

            x_train.extend(config_s[:self.n_train_tr_per_config])
            x_val.extend(config_s[self.n_train_tr_per_config:])

            y_train.extend(config_l[:self.n_train_tr_per_config])
            y_val.extend(config_l[self.n_train_tr_per_config:])

            pbs_train.extend(config_p[:self.n_train_tr_per_config])
            pbs_val.extend(config_p[self.n_train_tr_per_config:])

            tkbs_train.extend(config_k[:self.n_train_tr_per_config])
            tkbs_val.extend(config_k[self.n_train_tr_per_config:])

        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        pbs_train = np.vstack(pbs_train) 
        tkbs_train = np.vstack(tkbs_train)

        x_val = np.vstack(x_val)
        y_val = np.vstack(y_val)
        pbs_val = np.vstack(pbs_val)
        tkbs_val = np.vstack(tkbs_val)

        x_train, y_train, pbs_train, tkbs_train = self._shuffle(x_train, y_train, pbs_train, tkbs_train)
        x_val, y_val, pbs_val, tkbs_val = self._shuffle(x_val, y_val, pbs_val, tkbs_val)

        if self.out_mode == 'binary':
            y_train = np.unpackbits(y_train.astype(np.uint8)).reshape(-1, 8)
            y_val = np.unpackbits(y_val.astype(np.uint8)).reshape(-1, 8)

        train_data = (x_train, y_train, pbs_train, tkbs_train)
        val_data = (x_val, y_val, pbs_val, tkbs_val)

        return train_data, val_data

class TestDataLoader(DataLoader):
    def __init__(self, trace_files, tot_traces, target, byte_idx=None, out_mode='categorical', delta=0, op="SUM", sbox=None):
        super().__init__(trace_files, tot_traces, target, byte_idx, out_mode, delta, op, sbox)

    def load(self):
        x_test, y_test, pbs_test, tkbs_test = [], [], [], []

        for tfile in self.trace_files:
            data = np.load(tfile)
            TRXS = data['X_test']
            ptxs = data['ptxs_test']
            KEYS = data['keys_test'] 

            TRXS = process_traces_vectorized(TRXS, self.delta, self.operation)

            X_sub = TRXS[:self.n_tr_per_config]
            p_sub = ptxs[:self.n_tr_per_config]
            k_sub = KEYS[:self.n_tr_per_config] 

            config_s, config_l, config_p, config_k = [], [], [], []

            for i, x in enumerate(X_sub):
                pt = p_sub[i]
                current_key = k_sub[i] 
                l = aes.labels_from_key(pt, current_key, self.target, sbox=self.sbox) # <--- Uso S-Box dinamica

                if self.byte_idx is not None:
                    l = l[self.byte_idx]
                    pt = pt[self.byte_idx]
                    k_val = current_key[self.byte_idx]
                else:
                    k_val = current_key

                if self.out_mode == 'categorical':
                    l = to_categorical(l, self.n_classes)

                config_s.append(x)
                config_p.append(pt)
                config_k.append(k_val)
                config_l.append(l)

            x_test.extend(config_s)
            y_test.extend(config_l)
            pbs_test.extend(config_p)
            tkbs_test.extend(config_k)

        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        pbs_test = np.vstack(pbs_test) 
        tkbs_test = np.vstack(tkbs_test)

        x_test, y_test, pbs_test, tkbs_test = self._shuffle(x_test, y_test, pbs_test, tkbs_test)

        if self.out_mode == 'binary':
            y_test = np.unpackbits(y_test.astype(np.uint8)).reshape(-1, 8)

        return x_test, y_test, pbs_test, tkbs_test
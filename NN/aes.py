# Basics
import numpy as np

# Custom
from helpers import to_coords
import sys
sys.path.insert(0, '../utils')

import constants

# Typing
from typing import Callable

def _key(plaintext: np.ndarray, key: np.ndarray, sbox: np.ndarray = None):
    return key

def _sbox_in(plaintext: np.ndarray, key: np.ndarray, sbox: np.ndarray = None):
    return plaintext ^ key

def _sbox_out(plaintext: np.ndarray, key: np.ndarray, sbox: np.ndarray = None):
    sbox_in = plaintext ^ key
    if sbox is None:
        raise ValueError("SBOX parameter is required for SBOX_OUT target")
    return np.take(sbox, sbox_in)

def _hw_sbox_out(plaintext: np.ndarray, key: np.ndarray, sbox: np.ndarray = None):
    sbox_out = _sbox_out(plaintext, key, sbox)
    hw = [int(val).bit_count() for val in sbox_out] 
    return hw

def labels_from_key(plaintext: np.ndarray, key: np.ndarray, target: str, sbox: np.ndarray = None):
    actions = {
        'KEY': _key,
        'SBOX_IN': _sbox_in,
        'SBOX_OUT': _sbox_out,
        'HW_SO': _hw_sbox_out
    }
    generate_labels = actions[target] 
    labels = generate_labels(plaintext, key, sbox)
    return labels

def key_from_labels(ptx_byte, target: str, sbox: np.ndarray = None) -> np.ndarray:
    """
    Recovers the key relative to each possible value of the attack target.
    """
    possible_values = np.arange(256)

    if target == 'SBOX_IN': 
        sbox_in = possible_values
    elif target == 'SBOX_OUT': 
        if sbox is None:
            raise ValueError("SBOX parameter is required for SBOX_OUT target")
        # Calcolo dinamico e istantaneo dell'inversa della S-Box
        inv_sbox = np.zeros(256, dtype=int)
        inv_sbox[sbox] = np.arange(256)
        sbox_in = inv_sbox[possible_values]
    else:
        sbox_in = possible_values

    # Inverse-AddRoundKey
    key_bytes = np.array(sbox_in ^ ptx_byte)

    return key_bytes
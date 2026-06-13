import numpy as np


def process_traces_vectorized(trxs, delta, operation):
    # Se il delta è 0, non facciamo alcuno shift/somma e ritorniamo le tracce intatte
    if delta == 0:
        return trxs

    # 1. Creiamo due "viste" shiftate della matrice intera.
    train_T1 = trxs[:, :-delta]
    train_T2 = trxs[:, delta:]



    if operation == 'SUM':
        new_trxs = train_T1 + train_T2
    elif operation == 'ABS_SUB':
        new_trxs = abs(train_T1 - train_T2)
    elif operation == 'MUL':
        new_trxs = train_T1 * train_T2
    elif operation == 'DIV':
        # Per evitare divisioni per zero, aggiungiamo un piccolo epsilon
        epsilon = 1e-10
        new_trxs = train_T1 / (train_T2 + epsilon)
    elif operation == 'SUB':
        new_trxs = train_T1 - train_T2
    else:
        raise ValueError("Operation not supported")

    return new_trxs
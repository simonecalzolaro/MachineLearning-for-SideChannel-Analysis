# FEATURE REDUCTIONS FOR SBOX OUT
import os
import sys
import numpy as np
from sklearn.decomposition import PCA as SKCPA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def PCA(X_train, X_test, n_comp):
    pca = SKCPA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def LDA(X_train, X_test, ptxs_train, key_profile, target_byte, n_comp,SBOX):
    # --- FIX VARIABLE KEY (VETTORIALIZZATO) ---
    # Calcoliamo lo XOR per tutte le tracce contemporaneamente
    # key_profile è (N, 16), ptxs_train è (N, 16)
    labels_indices = ptxs_train[:, target_byte] ^ key_profile[:, target_byte]
    y_train = np.take(SBOX, labels_indices)
    
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda, lda
        

# FEATURE REDUCTIONS FOR HW
def get_hw(val):
    return bin(val).count('1')

HW_TABLE = np.array([get_hw(i) for i in range(256)])

def PCA_HW(X_train, X_test, n_comp):
    pca = SKCPA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def LDA_HW(X_train, X_test, ptxs_train, key_profile, target_byte, n_comp,SBOX):
    # --- FIX VARIABLE KEY (VETTORIALIZZATO) ---
    labels_indices = ptxs_train[:, target_byte] ^ key_profile[:, target_byte]
    sbox_out = np.take(SBOX, labels_indices)
    y_train = np.take(HW_TABLE, sbox_out)
    
    max_comp = len(np.unique(y_train)) - 1
    lda = LinearDiscriminantAnalysis(n_components=min(n_comp, max_comp))
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda, lda
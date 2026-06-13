import numpy as np
import aes

def min_att_tr(ge):
    """
    Returns the minimum number of traces required to reach GE = 0.
    """
    if len(np.where(ge == 0)[0]) > 0:
        return np.where(ge == 0)[0][0]
    else:
        return len(ge) + 1

def compute_final_rankings(predictions, plaintexts, target, sbox=None):
    """
    Computes the Guessing Entropy rankings using Forward Computation.
    Works flawlessly for both SBOX_OUT (256 classes) and HW_SO (9 classes).
    """
    # Prevent underflow for the logarithm
    predictions = np.clip(predictions, 1e-36, 1.0)
    log_probs = np.log(predictions)

    n_traces = len(predictions)
    
    # We test all 256 possible key guesses simultaneously
    all_key_guesses = np.arange(256)
    
    # For each plaintext, calculate the expected label (0-255 or 0-8) for EVERY possible key
    # expected_labels shape will be: (n_traces, 256)
    expected_labels = [aes.labels_from_key(pb, all_key_guesses, target, sbox) for pb in plaintexts]
    expected_labels = np.array(expected_labels, dtype=int)

    score = np.zeros(256)
    final_rankings = []

    for i in range(n_traces):
        # Extract the network's predicted log probability for the expected labels
        # If target is HW_SO, expected_labels[i] contains values 0-8.
        # log_probs[i] has 9 columns. The mapping aligns perfectly without crashing.
        trace_scores = log_probs[i, expected_labels[i]]
        
        # Add the probabilities to the running total for the 256 keys
        score += trace_scores
        
        # Rank the scores (descending order)
        ranking = np.argsort(score)[::-1]
        final_rankings.append(list(ranking))

    return final_rankings
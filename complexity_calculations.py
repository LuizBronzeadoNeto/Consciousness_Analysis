from numba import jit
import numpy

@jit(nopython=True)
def lz_fast(binary):
    n = len(binary)
    i, k, l = 0, 1, 1
    c = 1
    
    while True:
        if l + k - 1 >= n:
            c += 1
            break

        if binary[i + k - 1] == binary[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > 1:
                i += 1
                k -= 1
            else:
                c += 1
                l += 1
                if l > n:
                    break
                i = 0
                k = 1
    return c * numpy.log2(n) / n

import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

def encode_eeg_dynamic(signal, n_symbols):
    """
    Janela o sinal dinamicamente usando N símbolos (n_symbols deve ser par).
    Suporta até 36 símbolos através da string ALPHABET.
    """
    n_bins = n_symbols // 2
    diff = np.insert(np.diff(signal), 0, 0)
    signal_clipped = np.clip(signal, -10, 40)
    
    bins = np.linspace(-10, 40, n_bins + 1)
    
    encoded = []
    for v, d in zip(signal_clipped, diff):
        idx = np.digitize(v, bins) - 1
        idx = min(max(idx, 0), n_bins - 1)
        
        if d >= 0:  
            encoded.append(ALPHABET[idx]) 
        else:       
            neg_idx = (n_bins - 1) - idx
            encoded.append(ALPHABET[n_bins + neg_idx])
            
    return "".join(encoded)

def normalize_trace_dynamic(s, n_symbols):
    """
    Gera as regras de comutatividade dinamicamente e as aplica.
    """
    n_bins = n_symbols // 2
    prev_s = ""
    
    pairs = []
    for i in range(n_bins):
        sym_pos = ALPHABET[i]
        sym_neg = ALPHABET[(2 * n_bins - 1) - i]
        pairs.append((sym_neg + sym_pos, sym_pos + sym_neg))
        
    while s != prev_s:
        prev_s = s
        for p_in, p_out in pairs:
            s = s.replace(p_in, p_out)
    return s

def polz_compress_dynamic(encoded_string, n_symbols):
    dictionary = set() 
    w = ""             
    c = 0              
    
    for symbol in encoded_string:   
        candidate = normalize_trace_dynamic(w + symbol, n_symbols)
        if candidate in dictionary:
            w = candidate
        else:
            dictionary.add(candidate)
            c += 1
            w = ""
            
    if w != "":
        c += 1
        
    return c

def polz_complexity(signal, n_symbols):
    encoded_string = encode_eeg_dynamic(signal, n_symbols)
    n = len(encoded_string)
    if n == 0:
        return 0.0
    c = polz_compress_dynamic(encoded_string, n_symbols)
    return c * np.log2(n) / n

def lz_classic_binary(signal):
    """Lempel-Ziv clássico baseado na binarização pela mediana."""
    med = np.median(signal)
    binary = (signal >= med).astype(np.int8)
    n = len(binary)
    if n == 0:
        return 0.0
    c = lz_fast(binary)
    return c * np.log2(n) / n

def _chaos_batch(signal, cs):
    signal = numpy.asarray(signal, dtype=numpy.float64)
    cs = numpy.asarray(cs, dtype=numpy.float64)
    n = signal.size
    j = numpy.arange(n, dtype=numpy.float64)

    phase = j[:, None] * cs[None, :]
    cos_mat = numpy.cos(phase)
    sin_mat = numpy.sin(phase)

    weighted = signal[:, None]
    pc = numpy.cumsum(weighted * cos_mat, axis=0)
    qc = numpy.cumsum(weighted * sin_mat, axis=0)
    M = pc * pc + qc * qc

    E_x = signal.mean()
    denom = 1.0 - numpy.cos(cs)
    V_osc = (E_x * E_x) * (1.0 - cos_mat) / denom[None, :]
    D = M - V_osc

    jc = j - j.mean()
    j_var = (jc * jc).sum()
    Dc = D - D.mean(axis=0)
    D_var = (Dc * Dc).sum(axis=0)
    return (jc[:, None] * Dc).sum(axis=0) / numpy.sqrt(j_var * D_var)


def gottwald_melbourne_chaos(signal, c=None):
    if c is None:
        c = numpy.random.uniform(numpy.pi/5, 4*numpy.pi/5)
    return float(_chaos_batch(signal, numpy.array([c], dtype=numpy.float64))[0])

def median_K(signal, n_trials=50, seed=42):
    rng = numpy.random.RandomState(seed)
    cs = rng.uniform(numpy.pi/5, 4*numpy.pi/5, size=n_trials)
    return float(numpy.median(_chaos_batch(signal, cs)))

def criticality_proximity(k, alpha=0.85):
    k = numpy.clip(k, 0, 1)
    
    c = numpy.where(
        k < alpha,
        k / alpha,
        1 - (k - alpha) / (1 - alpha)
    )
    return c

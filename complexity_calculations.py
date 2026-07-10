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

def encode_eeg_to_trace(signal):
    """
    janela o sinal de EEG contínuo em uma string baseada na amplitude
    e na variação (derivada discreta) entre pontos.
    """
    # diferença para saber se está crescendo ou decrescendo
    diff = np.insert(np.diff(signal), 0, 0)
    
    signal_clipped = np.clip(signal, -10, 40)
    
    encoded = []
    for v, d in zip(signal_clipped, diff):
        if d >= 0:  # Crescendo ou estável
            if   -10 <= v < 0:  encoded.append('A')
            elif   0 <= v < 10: encoded.append('B')
            elif  10 <= v < 20: encoded.append('C')
            elif  20 <= v < 30: encoded.append('D')
            else:               encoded.append('E') 
        else:       # Decrescendo
            if    30 < v <= 40: encoded.append('F')
            elif  20 < v <= 30: encoded.append('G')
            elif  10 < v <= 20: encoded.append('H')
            elif   0 < v <= 10: encoded.append('I')
            else:               encoded.append('J') 
            
    return "".join(encoded)

def normalize_trace(s):
    """
    Normaliza a string para a forma canônica do monóide de traços.
    Comutatividades assumidas: AJ=JA, BI=IB, CH=HC, DG=GD, EF=FE.
    Como os pares são disjuntos, basta ordenar as ocorrências invertidas.
    """
    prev_s = ""
    while s != prev_s:
        prev_s = s
        s = s.replace('JA', 'AJ')
        s = s.replace('IB', 'BI')
        s = s.replace('HC', 'CH')
        s = s.replace('GD', 'DG')
        s = s.replace('FE', 'EF')
    return s

def polz_compress(encoded_string):
    """
    encoded_string: é o ômega, a string completa
    """
    dictionary = set() # D
    w = ""             # buffer tau τ
    c = 0              # contador
    
    for symbol in encoded_string:   # symbol é o sigma

        # juntamos o buffer com a letra nova (τ + σ)
        # normalize_trace --> relação de Independência (I)
        candidate = normalize_trace(w + symbol)
        
        if candidate in dictionary:
            w = candidate
        else:
            dictionary.add(candidate)
            c += 1
            w = ""
            
    # se o buffer acabar e ainda tiver algo no buffer
    if w != "":
        c += 1
        
    return c

def lempel_ziv_complexity(signal):
    """
    função principal
    """
    encoded_string = encode_eeg_to_trace(signal)
    
    n = len(encoded_string)
    if n == 0:
        return 0.0
        
    c = polz_compress(encoded_string)
    
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

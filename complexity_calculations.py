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

def lempel_ziv_complexity(signal):
    median = numpy.median(signal)
    binary = (signal > median).astype(numpy.int8)
    return lz_fast(binary)

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

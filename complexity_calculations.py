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

def gottwald_melbourne_chaos(signal, c=None):
    n = len(signal)
    if c is None:
        c = numpy.random.uniform(numpy.pi/5, 4*numpy.pi/5)
    j = numpy.arange(n)
    pc = numpy.cumsum(signal * numpy.cos(j*c))
    qc = numpy.cumsum(signal * numpy.sin(j*c))

    M = pc**2 + qc**2
    E_x = numpy.mean(signal)
    V_osc = E_x**2 * (1 - numpy.cos(j * c)) / (1 - numpy.cos(c))
    D = M - V_osc
    K = numpy.corrcoef(j, D)[0, 1]

    return K

def median_K(signal, n_trials=50, seed=42):
    rng = numpy.random.RandomState(seed)
    cs = rng.uniform(numpy.pi/5, 4*numpy.pi/5, size=n_trials)
    Ks = [gottwald_melbourne_chaos(signal, c=c) for c in cs]
    return numpy.median(Ks)

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
    return c / n

def lempel_ziv_complexity(signal):
    median = numpy.median(signal)
    binary = (signal > median).astype(numpy.int8)
    return lz_fast(binary)

def gottwald_melbourne_chaos(signal, c=None):
    n = len(signal)
    if c is None:
        c = numpy.random.uniform(0, numpy.pi)
    j = numpy.arange(n)
    pc = numpy.cumsum(signal * numpy.cos(j*c))
    qc = numpy.cumsum(signal * numpy.sin(j*c))

    M = pc**2 + qc**2
    K = numpy.corrcoef(j, M)[0, 1]

    return K

def median_K(signal, n_trials=50):
    Ks = []
    for _ in range(n_trials):
        Ks.append(gottwald_melbourne_chaos(signal))
    return numpy.median(Ks)

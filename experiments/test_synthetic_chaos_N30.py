import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

"""
REVIEW: 
Conduzir experimentos controlados com sinais simulados
 (periódicos, sistemas caóticos conhecidos, ruído branco/colorido, 
 ruído caótico e sequências autocorrelacionadas similares às séries de 
 potência). Demonstrar se a estatística K com apenas 30 observações 
 consegue separar caos determinístico de ruído estocástico.
"""

# ==============================================================================
# IMPLEMENTAÇÃO DO ALGORITMO 0-1 PARA CAOS (ESTATÍSTICA K)
# ==============================================================================
def test_01_chaos(series, c_vals=None, n_c=100):
    """
    Calcula a estatística K do Algoritmo 0-1 para Caos (Gottwald & Melbourne).
    Para séries muito curtas (N=30), usa-se a mediana de K sobre múltiplos c.
    """
    N = len(series)
    if c_vals is None:
        # Seleciona frequências c aleatórias no intervalo (pi/5, 4*pi/5)
        np.random.seed(42)
        c_vals = np.random.uniform(np.pi / 5, 4 * np.pi / 5, n_c)
    
    # Limite máximo de n para regressão assintótica (tipicamente N/10 em séries longas,
    # em N=30 usamos até N/3 para capturar dinâmica)
    max_n = max(3, N // 3)
    n_arr = np.arange(1, max_n + 1)
    
    Kc_list = []
    
    for c in c_vals:
        p = np.cumsum(series * np.cos(np.arange(1, N + 1) * c))
        q = np.cumsum(series * np.sin(np.arange(1, N + 1) * c))
        
        # Deslocamento Quadrático Médio M(n)
        M = np.zeros(max_n)
        for n in n_arr:
            disp_p = (p[n:] - p[:-n])**2
            disp_q = (q[n:] - q[:-n])**2
            M[n-1] = np.mean(disp_p + disp_q)
        
        # Termo de modificação de covariância D(n)
        f_c = np.mean(series) * (1 - np.cos(n_arr * c)) / (1 - np.cos(c))
        D = M - f_c
        
        # Coeficiente de correlação logarítmica / de postos (K)
        if np.std(D) > 1e-12:
            r = np.corrcoef(n_arr, D)[0, 1]
            if not np.isnan(r):
                Kc_list.append(r)
    
    if len(Kc_list) == 0:
        return 0.0
    
    # Mediana para neutralizar ressonâncias em valores específicos de c
    K = np.median(Kc_list)
    return np.clip(K, 0.0, 1.0)

# ==============================================================================
# GERADORES DE SINAIS SINTÉTICOS (N = 30)
# ==============================================================================
def generate_signals(N=30):
    t = np.arange(N)
    
    # 1. Periódico / Multiperiódico
    sig_periodic = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.23 * t)
    
    # 2. Caos Determinístico (Mapa Logístico r=4.0)
    sig_chaos = np.zeros(N)
    x = np.random.uniform(0.1, 0.9)
    for i in range(100): x = 4.0 * x * (1 - x) # Warm-up para transiente
    for i in range(N):
        x = 4.0 * x * (1 - x)
        sig_chaos[i] = x
        
    # 3. Ruído Estocástico Puro (Gaussiano / Branco)
    sig_noise = np.random.normal(0, 1, N)
    
    # 4. Caos Contaminado com Ruído (SNR = 10dB)
    noise_component = np.random.normal(0, 0.3, N)
    sig_chaotic_noise = sig_chaos + noise_component
    
    # 5. Série Autocorrelacionada (Processo AR(1) com phi=0.8)
    sig_ar1 = np.zeros(N)
    sig_ar1[0] = np.random.normal(0, 1)
    for i in range(1, N):
        sig_ar1[i] = 0.8 * sig_ar1[i-1] + np.random.normal(0, 0.6)
        
    return {
        "Periódico": sig_periodic,
        "Caos Puro": sig_chaos,
        "Ruído Branco": sig_noise,
        "Caos + Ruído": sig_chaotic_noise,
        "AR(1) Autocorrelacionado": sig_ar1
    }

# ==============================================================================
# EXPERIMENTO MONTE CARLO E AVALIAÇÃO DE SEPARABILIDADE
# ==============================================================================
def run_experiment(N=30, M=200):
    print(f"Iniciando simulação Monte Carlo: M={M} séries de tamanho N={N}...")
    
    results = {
        "Periódico": [],
        "Caos Puro": [],
        "Ruído Branco": [],
        "Caos + Ruído": [],
        "AR(1) Autocorrelacionado": []
    }
    
    for _ in range(M):
        signals = generate_signals(N=N)
        for key, sig in signals.items():
            k_val = test_01_chaos(sig)
            results[key].append(k_val)
            
    # resumo
    print("\n" + "="*60)
    print(f"{'Classe de Sinal':<25} | {'K Médio':<10} | {'Desv. Padrão':<10}")
    print("="*60)
    for key, val in results.items():
        print(f"{key:<25} | {np.mean(val):.4f}     | {np.std(val):.4f}")
    print("="*60)
    
    # Caos Puro (1) vs Ruído Branco (0)
    y_true = np.array([1]*M + [0]*M)
    y_scores = np.array(results["Caos Puro"] + results["Ruído Branco"])
    auc_caos_vs_ruido = roc_auc_score(y_true, y_scores)
    
    # Caos Puro (1) vs AR(1) (0)
    y_scores_ar1 = np.array(results["Caos Puro"] + results["AR(1) Autocorrelacionado"])
    auc_caos_vs_ar1 = roc_auc_score(y_true, y_scores_ar1)
    
    print(f"\nAUC (Caos Puro vs Ruído Branco) [N={N}]: {auc_caos_vs_ruido:.4f}")
    print(f"AUC (Caos Puro vs AR(1) Autocorrelacionado) [N={N}]: {auc_caos_vs_ar1:.4f}")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    results = run_experiment(N=30, M=200)
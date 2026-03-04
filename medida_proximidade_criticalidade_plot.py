import numpy as np
import matplotlib.pyplot as plt
import complexity_calculations as eeg

def plot_c_function(alpha=0.85):
    k_values = np.linspace(0, 1, 500)
    
    c_values = [eeg.medida_proximidade_criticalidade(k, alpha) for k in k_values]

    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, c_values, color='red', linewidth=2.5, label=f'Curva de Criticalidade (alpha={alpha}$)')
    
    plt.axvline(x=alpha, color='gray', linestyle='--', alpha=0.6)
    plt.scatter([alpha], [1.0], color='black', zorder=5)
    plt.text(alpha, 1.02, 'Borda do Caos\n(C=1)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Medida de Proximidade à Criticalidade (C)', fontsize=14)
    plt.xlabel('K', fontsize=12)
    plt.ylabel('C', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    plot_c_function(alpha=0.85)
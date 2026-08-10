import numpy as np
import complexity_calculations as cc

def executar_teste():
    print("--- INÍCIO DO TESTE POLZ ---")
    
    # EEG simulado
    # 100 pontos de uma onda sinusoidal com ruído, 
    # escalada para variar dentro da nossa faixa de -10 a 40 dB.
    tempo = np.linspace(0, 10, 100)
    sinal_simulado = 15 + 20 * np.sin(tempo) + np.random.normal(0, 2.5, 100)
    
    # codificação da Trajetória (String)
    string_codificada = cc.encode_eeg_to_trace(sinal_simulado)
    print("\n1. SINAL CODIFICADO (Bruto):")
    print(string_codificada)
    print(f"Tamanho da string gerada: {len(string_codificada)} caracteres")
    
    string_comutativa = "AJAJJAIBBI"
    string_normalizada = cc.normalize_trace(string_comutativa)
    print("\n2. TESTE DE NORMALIZAÇÃO (COMUTATIVIDADE):")
    print(f"String Original:   {string_comutativa}")
    print(f"String Normalizada:{string_normalizada}")
    if "JA" not in string_normalizada and "IB" not in string_normalizada:
         print("-> Sucesso: Os pares comutativos foram normalizados corretamente!")
    
    num_frases = cc.polz_compress(string_codificada)
    complexidade_final = cc.lempel_ziv_complexity(sinal_simulado)
    
    print("\n3. RESULTADO DA COMPLEXIDADE:")
    print(f"Número de frases no dicionário POLZ (C): {num_frases}")
    print(f"Complexidade Lempel-Ziv normalizada: {complexidade_final:.4f}")
    print("----------------------------\n")

if __name__ == "__main__":
    executar_teste()
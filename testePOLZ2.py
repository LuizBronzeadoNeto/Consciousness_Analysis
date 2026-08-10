import numpy as np
import complexity_calculations as cc

def executar_testes_extremos():
    print("--- TESTES DE ESTRESSE POLZ ---\n")
    
    # Simula um sinal "zerado" (ex: 0 dB constante) com 50 pontos
    sinal_plano = np.zeros(50)
    str_plano = cc.encode_eeg_to_trace(sinal_plano)
    c_plano = cc.lempel_ziv_complexity(sinal_plano)
    
    print("1. TESTE DE SINAL CONSTANTE (Linha de base):")
    print(f"String gerada: {str_plano[:20]}... (Tamanho total: {len(str_plano)})")
    print(f"Complexidade Normalizada: {c_plano:.4f}")
    print("Esperado: Uma string de letras iguais (ex: BBB...) e complexidade próxima de 0.\n")
    
    # Valores saltando caoticamente de -50 a 100 dB (além da faixa -10 a 40)
    sinal_caos = np.random.uniform(-50, 100, 100)
    str_caos = cc.encode_eeg_to_trace(sinal_caos)
    c_caos = cc.lempel_ziv_complexity(sinal_caos)
    
    print("2. TESTE DE RUÍDO EXTREMO E CLIPPING:")
    print(f"String gerada (amostra): {str_caos[:30]}...")
    print(f"Complexidade Normalizada: {c_caos:.4f}")
    print("Esperado: O programa não pode dar erro. Letras limitadas a A e J (ou E e F) nas extremidades, e alta complexidade.\n")
    
    sinal_alt = np.array([5, -5] * 25)
    str_alt = cc.encode_eeg_to_trace(sinal_alt)
    c_alt = cc.lempel_ziv_complexity(sinal_alt)
    
    print("3. TESTE DE PADRÃO ALTERNANTE ESTRITO:")
    print(f"String gerada (amostra): {str_alt[:30]}...")
    print(f"Complexidade Normalizada: {c_alt:.4f}")
    print("Esperado: Uma repetição rítmica (ex: BJBJBJ...) com complexidade moderadamente baixa (abaixo do caos).\n")
    
    print("--- FIM DOS TESTES ---")

if __name__ == "__main__":
    executar_testes_extremos()
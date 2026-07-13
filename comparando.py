import pandas as pd
import numpy as np
import complexity_calculations as cc

def comparar_metodos():
    caminho_arquivo = "dataset/Volunteer/02_Sdb.csv"
    print(f"Lendo o arquivo: {caminho_arquivo}...\n")
    
    try:
        df = pd.read_csv(caminho_arquivo, header=None)
        
        sinal_real = df.iloc[0, 10:30].values 
        
        mediana = np.median(sinal_real)
        binario_antigo = (sinal_real > mediana).astype(np.int8)
        string_antiga = "".join(map(str, binario_antigo))
        
        diff_array = np.insert(np.diff(sinal_real), 0, 0) # Derivada discreta
        string_nova = cc.encode_eeg_to_trace(sinal_real)
        
        print("=== COMPARAÇÃO DA DISCRETIZAÇÃO DO SINAL ===\n")
        print(f"Mediana calculada para a janela: {mediana:.2f} dB")
        print("-" * 75)
        print(f"{'Ponto':<6} | {'Sinal (dB)':<12} | {'Variação (Δ)':<14} || {'ANTES (0 ou 1)':<15} | {'DEPOIS (A a J)':<15}")
        print("-" * 75)
        
        for i in range(len(sinal_real)):
            val = sinal_real[i]
            delta = diff_array[i]
            antigo = string_antiga[i]
            novo = string_nova[i]
            
            delta_str = f"{delta:+.2f}" if i > 0 else "0.00"
            direcao = "Crescendo/Estável" if delta >= 0 else "Decrescendo"
            
            print(f"{i:<6} | {val:>9.2f} dB | {delta_str:>7}  || {antigo:>9}       | {novo:>8} ")
            
        print("-" * 75)
        print("\n=== RESUMO DAS STRINGS GERADAS ===")
        print(f"String ANTIGA (LZ Clássico):  {string_antiga}")
        print(f"String NOVA   (POLZ):         {string_nova}")
        print("==================================\n")
        
    except FileNotFoundError:
        print("Arquivo não encontrado. Execute o script na raiz do projeto 'CONSCIOUSNESS_ANALYSIS'.")

if __name__ == "__main__":
    comparar_metodos()
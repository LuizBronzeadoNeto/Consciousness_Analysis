import pandas as pd
import numpy as np
import complexity_calculations as cc

def debug_polz_interativo(encoded_string):
   
    dictionary = set()
    w = ""
    c = 0
    
    print("==================================================")
    print(f"STRING A SER COMPRIMIDA: {encoded_string}")
    print(f"TAMANHO: {len(encoded_string)} letras")
    print("==================================================\n")
    
    for i, symbol in enumerate(encoded_string):
        print(f"--- Passo {i+1} ---")
        print(f"Lemos a letra: '{symbol}'")
        
        # juntamos o que estava no buffer com a nova letra
        candidate_raw = w + symbol
        
        # aplicar a regra da Ordem Parcial (Monóide)
        candidate_norm = cc.normalize_trace(candidate_raw)
        
        print(f"Buffer + Letra: '{w}' + '{symbol}' = '{candidate_raw}'")
        if candidate_raw != candidate_norm:
            print(f" -> Normalização (Ordem Parcial): '{candidate_raw}' foi reordenado para '{candidate_norm}'")
        else:
            print(f" -> Normalização: '{candidate_norm}' (já está na forma canônica)")
            
        # verificação no dicionário
        if candidate_norm in dictionary:
            print(f"A string '{candidate_norm}' ESTÁ no dicionário?")
            print(f" -> SIM! Já conhecemos essa dinâmica. Guardamos no buffer.")
            w = candidate_norm
        else:
            print(f"A string '{candidate_norm}' ESTÁ no dicionário?")
            print(f" -> NÃO. É um padrão novo! Adicionando ao dicionário...")
            dictionary.add(candidate_norm)
            c += 1
            w = "" # resetar o buffer
            
        # mostra o estado atual da memória
        print(f"[Estado] Frases (C) = {c} | Buffer = '{w}'")
        print(f"[Dicionário Atual] {dictionary}\n")
        
    print("==================================================")
    print(f"COMPRESSÃO FINALIZADA!")
    print(f"Total de sub-padrões únicos encontrados (C): {c}")
    print("==================================================")

def executar_com_dados_reais():
    # (Voluntário 02)
    caminho_arquivo = "dataset/Volunteer/02_Sdb.csv"
    print(f"Carregando dados de: {caminho_arquivo}...")
    
    try:
        df = pd.read_csv(caminho_arquivo, header=None)
        
        #1ª janela de tempo (linha 0), frequências de índice 10 a 25
        sinal_real = df.iloc[0, 10:25].values 
        
        print("\nSinal Original de Potência (dB) extraído:")
        print(np.round(sinal_real, 2))
        print("\nCodificando o sinal para o espaço de fase (A-J)...")
        
        
        string_codificada = cc.encode_eeg_to_trace(sinal_real)
        
        debug_polz_interativo(string_codificada)
        
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado no caminho {caminho_arquivo}.")
        print("Certifique-se de rodar este script a partir da raiz do projeto.")

if __name__ == "__main__":
    executar_com_dados_reais()
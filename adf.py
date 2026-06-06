import numpy as np
import pandas as pd
from utility import custom_adf, periodograma

ALPHA = 0.05 # Nivel de significancia estadístico

def diferenciar_serie(y, d, D, s):
    w = y.copy()
    for _ in range(D):
        w = w[s:] - w[:-s]
    for _ in range(d):
        w = w[1:] - w[:-1]
    return w

def main():
    print("-> Iniciando adf.py (100% Dinámico)...")
    
    try:
        df = pd.read_csv('tserie.csv')
        y = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
    except FileNotFoundError:
        print("[Error] No se encontró 'tserie.csv'.")
        return

    # --- Detección Automática del Ciclo Estacional (s) ---
    print("-> Detectando ciclo estacional mediante Periodograma...")
    f_bins, I_f = periodograma(y)
    
    # Ignorar la componente DC (frecuencia 0)
    f_bins, I_f = f_bins[1:], I_f[1:]
    
    # Encontrar frecuencia dominante
    idx_max = np.argmax(I_f)
    f_dom = f_bins[idx_max]
    ciclo_s_auto = int(np.round(1.0 / f_dom)) if f_dom > 0 else 1
    print(f"   [!] Frecuencia dominante: {f_dom:.4f} Hz -> Ciclo 's' detectado: {ciclo_s_auto} muestras.")

    mejor_d, mejor_D = 0, 0
    estacionaria = False
    orden_total = 0

    print("-> Buscando integraciones d y D (Expansión sin techo)...")
    while not estacionaria:
        combinaciones = [(d_test, orden_total - d_test) for d_test in range(orden_total + 1)]
            
        for d_test, D_test in combinaciones:
            serie_diff = diferenciar_serie(y, d_test, D_test, ciclo_s_auto)
            
            if len(serie_diff) < 20: continue
                
            t_stat, crit_val, es_estacionaria = custom_adf(serie_diff, alpha=ALPHA)
            print(f"   Orden {orden_total} | (d={d_test}, D={D_test}): t={t_stat:.4f}, crit={crit_val:.4f} -> Estacionaria: {es_estacionaria}")
            
            if es_estacionaria:
                mejor_d, mejor_D = d_test, D_test
                estacionaria = True
                break

        if estacionaria: break
        orden_total += 1
        
        if len(y) - (orden_total * max(1, ciclo_s_auto)) < 20:
            print("-> [Aviso] Límite de datos alcanzado. Seleccionando la última válida.")
            mejor_d, mejor_D = orden_total - 1, 0
            break

    adf_config = pd.DataFrame({'d': [mejor_d], 'D': [mejor_D], 's': [ciclo_s_auto], 'alpha': [ALPHA]})
    adf_config.to_csv('adf.csv', index=False)
    
    serie_optima = diferenciar_serie(y, mejor_d, mejor_D, ciclo_s_auto)
    pd.DataFrame(serie_optima, columns=['w_t']).to_csv('serie_diferenciada.csv', index=False)
    print(f"-> [OK] Fin ADF. d={mejor_d}, D={mejor_D}, s={ciclo_s_auto} guardados.")

if __name__ == "__main__":
    main()
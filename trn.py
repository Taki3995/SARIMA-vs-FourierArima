import numpy as np
import pandas as pd
from utility import pinv_svd, ols_estimate, calcular_aic, periodograma, es_ruido_blanco

# La regularización se deja en 0 (puro OLS) ya que SVD se encarga de la colinealidad naturalmente.
LAMBDA = 0.0  

def entrenar_farima_dinamico(y):
    print("   -> Analizando Periodograma para F-ARIMA...")
    f_bins, I_f = periodograma(y)
    f_bins, I_f = f_bins[1:], I_f[1:]
    
    # Filtro estadístico: Seleccionamos frecuencias cuya energía supere la media + 1 Desviación Estándar
    umbral = np.mean(I_f) + np.std(I_f)
    top_indices = np.where(I_f > umbral)[0]
    
    # Failsafe: Si ninguna supera el umbral estricto, tomamos la más alta
    if len(top_indices) == 0:
        top_indices = [np.argmax(I_f)]
        
    top_freqs = f_bins[top_indices]
    print(f"   -> {len(top_freqs)} armónicos detectados por encima del ruido de fondo.")
    
    N = len(y)
    t = np.arange(N)
    X_fourier = [np.ones(N)]
    for freq in top_freqs:
        X_fourier.append(np.cos(2 * np.pi * freq * t))
        X_fourier.append(np.sin(2 * np.pi * freq * t))
        
    X_fourier = np.column_stack(X_fourier)
    Gamma_hat = ols_estimate(X_fourier, y, lam=LAMBDA)
    
    return Gamma_hat, top_freqs

def auto_fase_I_innovaciones(w):
    N = len(w)
    max_ka_search = int(np.sqrt(N)) # Límite heurístico máximo seguro
    
    mejor_ka = 1
    mejor_aic = float('inf')
    mejor_eps = None
    
    print(f"   -> Auto-detectando memoria AR(Ka) desde 1 hasta {max_ka_search}...")
    for ka_test in range(1, max_ka_search + 1):
        Y = w[ka_test:]
        X = np.array([w[t-ka_test : t][::-1] for t in range(ka_test, N)])
        
        Gamma_ar = ols_estimate(X, Y, lam=LAMBDA)
        eps_estimado = Y - (X @ Gamma_ar)
        
        aic_actual = calcular_aic(eps_estimado, ka_test, len(Y))
        if aic_actual < mejor_aic:
            mejor_aic = aic_actual
            mejor_ka = ka_test
            mejor_eps = np.concatenate((np.zeros(ka_test), eps_estimado))
            
    print(f"   -> [!] Memoria óptima K_a detectada: {mejor_ka}")
    return mejor_eps, mejor_ka

def fase_II_busqueda_dinamica(w, eps, s, K_a):
    N = len(w)
    capa_actual = 0
    limite_paciencia = 4 # Hasta orden máximo de 4 por componente (asegura término en tiempo finito)
    
    modelos_evaluados = [] # Guardará tuplas: (aic, modelo, eta, residuos_e)
    
    while capa_actual <= limite_paciencia:
        print(f"   -> Explorando Capa de Complejidad {capa_actual}...")
        
        # Generar combinaciones pertenecientes EXACTAMENTE a esta capa
        # Una combinación es de esta capa si su orden máximo es exactamente 'capa_actual'
        for p in range(capa_actual + 1):
            for q in range(capa_actual + 1):
                for P in range(capa_actual + 1):
                    for Q in range(capa_actual + 1):
                        
                        if max(p, q, P, Q) != capa_actual:
                            continue # Ya se evaluó en una capa anterior
                            
                        if p == 0 and q == 0 and P == 0 and Q == 0:
                            continue
                            
                        max_lag = max(p, P * s, q, Q * s, K_a)
                        if max_lag >= N - 10: # Failsafe de datos insuficientes
                            continue 
                            
                        Y_target = w[max_lag:]
                        X_expandida = []
                        
                        for t in range(max_lag, N):
                            fila = [1.0]
                            for i in range(1, p + 1): fila.append(w[t - i])
                            for j in range(1, P + 1): fila.append(w[t - j * s])
                            for i in range(1, q + 1): fila.append(eps[t - i])
                            for j in range(1, Q + 1): fila.append(eps[t - j * s])
                            X_expandida.append(fila)
                            
                        X_expandida = np.array(X_expandida)
                        
                        # Manejo de matrices singulares o inestables en OLS
                        try:
                            eta_hat = ols_estimate(X_expandida, Y_target, lam=LAMBDA)
                            e_t = Y_target - (X_expandida @ eta_hat)
                            aic = calcular_aic(e_t, len(eta_hat), len(Y_target))
                            modelos_evaluados.append((aic, (p, q, P, Q), eta_hat, e_t))
                        except Exception:
                            pass # Ignorar modelo matemáticamente inestable
        
        # 1. Ordenar todos los modelos evaluados hasta ahora por AIC
        modelos_evaluados.sort(key=lambda x: x[0])
        
        # 2. El Filtro: Verificar si el MEJOR modelo (menor AIC) es Ruido Blanco
        mejor_aic_actual, mejor_modelo_actual, eta_actual, e_actual = modelos_evaluados[0]
        
        if es_ruido_blanco(e_actual):
            print(f"   -> [¡ÉXITO!] Convergencia alcanzada en Capa {capa_actual}.")
            print(f"   -> El modelo ganador tiene los residuos puros (Ruido Blanco).")
            return mejor_modelo_actual, eta_actual, mejor_aic_actual
            
        capa_actual += 1

    # Fallback: Si se agotó la paciencia y ningún residuo es Ruido Blanco perfecto,
    # buscamos en la lista el primero que sí lo sea, aunque su AIC no sea el absoluto menor.
    print("   -> [Aviso] Límite de expansión alcanzado. Buscando el mejor candidato válido...")
    for aic, modelo, eta, e in modelos_evaluados:
        if es_ruido_blanco(e):
            print("   -> [OK] Se encontró un modelo sub-óptimo en AIC pero válido en residuos.")
            return modelo, eta, aic
            
    # Último recurso: Devolver el de menor AIC advirtiendo correlación
    print("   -> [Advertencia CRÍTICA] Ningún modelo logró Ruido Blanco. Retornando el de mínimo AIC.")
    return modelos_evaluados[0][1], modelos_evaluados[0][2], modelos_evaluados[0][0]

def main():
    print("-> Iniciando trn.py (100% Dinámico y Autónomo)...")
    
    try:
        df_y = pd.read_csv('tserie.csv')
        y = df_y.iloc[:, 1].values if df_y.shape[1] > 1 else df_y.iloc[:, 0].values
        
        w = pd.read_csv('serie_diferenciada.csv')['w_t'].values
        s = int(pd.read_csv('adf.csv')['s'].iloc[0])
    except FileNotFoundError as e:
        print(f"[Error] Archivo faltante: {e}")
        return

    # --- F-ARIMA ---
    print("\n[Entrenando F-ARIMA]")
    gamma_f, freqs = entrenar_farima_dinamico(y)
    pd.DataFrame({'frecuencias': np.pad(freqs, (0, len(gamma_f) - len(freqs)), constant_values=np.nan),
                  'coeficientes': gamma_f}).to_csv('train_farima.csv', index=False)
                  
    # --- SARIMA ---
    print("\n[Entrenando SARIMA]")
    eps, K_a_auto = auto_fase_I_innovaciones(w)
    
    mejor_modelo, mejor_eta, mejor_aic = fase_II_busqueda_dinamica(w, eps, s, K_a_auto)
    p, q, P, Q = mejor_modelo
    
    print(f"\n=> RESULTADO SARIMA: Orden (p={p}, q={q})(P={P}, Q={Q})_s={s} | AIC: {mejor_aic:.4f}")
    
    df_sarima = pd.DataFrame({'p': [p], 'q': [q], 'P': [P], 'Q': [Q], 'AIC': [mejor_aic], 'K_a': [K_a_auto], 'Lambda': [LAMBDA]})
    for idx, coef in enumerate(mejor_eta):
        df_sarima[f'coef_{idx}'] = coef
    df_sarima.to_csv('train_sarima.csv', index=False)
    
if __name__ == "__main__":
    main()
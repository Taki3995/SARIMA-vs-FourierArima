import numpy as np
import pandas as pd
from utility import periodograma, estimar_gamma_farima, estimar_eta_sarima

# =============================================================================
# 1. FUNCIONES AUXILIARES PARA ENTRENAMIENTO
# =============================================================================

def calcular_aic(sse, N, num_params):
    """
    Calcula el AIC según la fórmula proporcionada:
    aic = log(sse) + (2(p+2)) / (N - p - 3)
    (Se usa num_params en lugar de 'p' para no confundir con el rezago AR)
    """
    denominador = N - num_params - 3
    if sse <= 0 or denominador <= 0:
        return np.inf # Penalización si no hay suficientes grados de libertad
        
    aic = np.log(sse) + (2 * (num_params + 2)) / denominador
    return aic

def diferenciar_serie(y, d, D, s):
    """Aplica diferenciación ordinaria y estacional: w_t = nabla^d nabla_s^D y_t"""
    w = np.array(y.copy())
    for _ in range(int(D)):
        w = w[s:] - w[:-s]
    for _ in range(int(d)):
        w = np.diff(w, n=1)
    return w

def construir_matriz_fourier(t_n, T_p, K_p):
    """Construye la matriz de diseño X para la serie de Fourier."""
    n_samples = len(t_n)
    X = np.zeros((n_samples, 2 * K_p))
    for k in range(1, K_p + 1):
        arg = (2 * np.pi * k * t_n) / T_p
        X[:, 2*(k-1)] = np.cos(arg)
        X[:, 2*(k-1) + 1] = np.sin(arg)
    return X

# =============================================================================
# 2. ALGORITMOS SARIMA (TWO-PHASE OLS)
# =============================================================================

def sarima_fase_1(w, K_a):
    """Phase I: Estimación de Residuos vía AR(K_a)"""
    n_samples = len(w) - K_a
    if n_samples <= 0:
        return None, None
        
    Y = w[K_a:]
    Z = np.zeros((n_samples, K_a))
    
    for t in range(n_samples):
        # z_t = [w_{t-1}, w_{t-2}, ..., w_{t-K_a}]
        Z[t, :] = w[t : t + K_a][::-1] 
        
    try:
        Gamma_hat = np.dot(np.dot(np.linalg.inv(np.dot(Z.T, Z)), Z.T), Y)
    except np.linalg.LinAlgError:
        return None, None
        
    epsilon_hat = Y - np.dot(Z, Gamma_hat)
    return Gamma_hat, epsilon_hat

def construir_conjuntos_retardo(p, q, P, Q, s):
    """Construye los conjuntos de rezagos L_A y L_M."""
    L_A = set([i for i in range(1, p + 1)])
    L_A.update([j * s for j in range(1, P + 1)])
    for i in range(1, p + 1):
        for j in range(1, P + 1):
            L_A.add(i + j * s)
            
    L_M = set([m for m in range(1, q + 1)])
    L_M.update([n * s for n in range(1, Q + 1)])
    for m in range(1, q + 1):
        for n in range(1, Q + 1):
            L_M.add(m + n * s)
            
    return sorted(list(L_A)), sorted(list(L_M))

def reconstruir_w_y_calcular_sse(w, epsilon_hat, K_a, L_A, L_M, eta_hat):
    """Evalúa la serie w_t ajustada con Fase II para obtener el SSE."""
    max_lag_A = max(L_A) if L_A else 0
    max_lag_M = max(L_M) if L_M else 0
    tau = max(K_a, max_lag_A, max_lag_M)
    
    if tau >= len(w):
        return np.inf, 0

    sse = 0.0
    n_eval = 0
    for t in range(tau, len(w)):
        x_A_t = [w[t - l] for l in L_A]
        x_M_t = []
        for l in L_M:
            idx_eps = t - l - K_a
            if idx_eps >= 0 and idx_eps < len(epsilon_hat):
                x_M_t.append(epsilon_hat[idx_eps])
            else:
                x_M_t.append(0.0)
                
        X_t = np.array(x_A_t + x_M_t)
        pred = np.dot(X_t, eta_hat)
        sse += (w[t] - pred)**2
        n_eval += 1
        
    return sse, n_eval

def grid_search_sarima(w, p_max, q_max, P_max, Q_max, s, K_a, lam):
    """Itera combinaciones minimizando AIC vía OLS."""
    Gamma_hat, epsilon_hat = sarima_fase_1(w, K_a)
    if Gamma_hat is None:
        return None
        
    best_aic = np.inf
    best_params = None
    best_eta = None
    best_L_A = None
    best_L_M = None
    
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            for P in range(P_max + 1):
                for Q in range(Q_max + 1):
                    if p == 0 and q == 0 and P == 0 and Q == 0:
                        continue
                        
                    L_A, L_M = construir_conjuntos_retardo(p, q, P, Q, s)
                    
                    max_lag_A = max(L_A) if L_A else 0
                    max_lag_M = max(L_M) if L_M else 0
                    tau = max(K_a, max_lag_A, max_lag_M)
                    
                    if tau >= len(w):
                        continue
                        
                    X_t_array = []
                    w_array = []
                    for t in range(tau, len(w)):
                        x_A_t = [w[t - l] for l in L_A]
                        x_M_t = []
                        for l in L_M:
                            idx_eps = t - l - K_a
                            if idx_eps >= 0 and idx_eps < len(epsilon_hat):
                                x_M_t.append(epsilon_hat[idx_eps])
                            else:
                                x_M_t.append(0.0)
                        X_t_array.append(x_A_t + x_M_t)
                        w_array.append(w[t])
                        
                    eta_hat = estimar_eta_sarima(X_t_array, w_array)
                    if eta_hat is None:
                        continue
                        
                    eta_hat = eta_hat.flatten()
                    sse, n_eval = reconstruir_w_y_calcular_sse(w, epsilon_hat, K_a, L_A, L_M, eta_hat)
                    
                    num_params = len(L_A) + len(L_M)
                    aic = calcular_aic(sse, n_eval, num_params)
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, q, P, Q)
                        best_eta = eta_hat
                        best_L_A = L_A
                        best_L_M = L_M
                        
    if best_params is None:
        return None
        
    return {
        'aic': float(best_aic),
        'order': best_params,
        'eta': best_eta.tolist(),
        'L_A': best_L_A,
        'L_M': best_L_M,
        'Gamma_hat': Gamma_hat.tolist() if Gamma_hat is not None else []
    }

# =============================================================================
# 3. ALGORITMOS F-ARIMA
# =============================================================================

def entrenar_farima(y, d, K_p_max, p_max, q_max, K_a, lam):
    """Entrena Fourier (Periodograma + OLS-Penalizado) y modela el residuo con ARIMA(p,d,q)"""
    f_k, I_fk = periodograma(y)
    
    # Excluir frecuencia 0 (media continua) para buscar la frecuencia fundamental
    I_fk[0] = 0
    idx_max = np.argmax(I_fk)
    f_max = f_k[idx_max]
    T_p = 1.0 / f_max if f_max > 0 else len(y)
    
    t_n = np.arange(len(y))
    
    best_aic_fourier = np.inf
    best_Kp = 1
    best_gamma = None
    best_residuals = None
    
    for K_p in range(1, K_p_max + 1):
        X = construir_matriz_fourier(t_n, T_p, K_p)
        gamma_hat = estimar_gamma_farima(X, y, lam).flatten()
        
        pred = np.dot(X, gamma_hat)
        res = y - pred
        
        sse = np.sum(res**2)
        n_eval = len(y)
        num_params = 2 * K_p
        
        aic = calcular_aic(sse, n_eval, num_params)
        
        if aic < best_aic_fourier:
            best_aic_fourier = aic
            best_Kp = K_p
            best_gamma = gamma_hat
            best_residuals = res
            
    # Componente ARIMA(p,d,q) sobre los residuos. Lógica SARIMA con D=0, s=0
    w_residuals = diferenciar_serie(best_residuals, d, 0, 0)
    arima_res = grid_search_sarima(w_residuals, p_max, q_max, 0, 0, 0, K_a, lam)
    
    return {
        'T_p': float(T_p),
        'K_p': best_Kp,
        'gamma': best_gamma.tolist() if best_gamma is not None else [],
        'arima_model': arima_res
    }

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    
    # 1. Hiperparámetros fijos de configuración para la búsqueda
    train_size = 0.8
    p_max = 2
    q_max = 2
    P_max = 1
    Q_max = 1
    K_a = 15
    lam = 0.01
    K_p_max = 5
    
    # 2. Cargar datos
    adf_results = pd.read_csv("adf.csv")
    d = int(adf_results['d'].values[0])
    D = int(adf_results['D'].values[0])
    s = int(adf_results['s'].values[0])
    
    datos = pd.read_csv("tserie.csv")
    y_full = datos.iloc[:, 0].values
    
    # Dividir Entrenamiento
    n_train = int(len(y_full) * train_size)
    y_train = y_full[:n_train]
    
    # 3. Entrenar SARIMA
    w_train = diferenciar_serie(y_train, d, D, s)
    modelo_sarima = grid_search_sarima(w_train, p_max, q_max, P_max, Q_max, s, K_a, lam)
    
    # 4. Entrenar F-ARIMA
    modelo_farima = entrenar_farima(y_train, d, K_p_max, p_max, q_max, K_a, lam)
    
    # 5. Guardar resultados en CSV usando solo Pandas (Sin Json)
    resultados = {
        'modelo': ['SARIMA', 'FARIMA'],
        'order': [str(modelo_sarima['order']) if modelo_sarima else '', 
                  str(modelo_farima['arima_model']['order']) if modelo_farima['arima_model'] else ''],
        'eta': [str(modelo_sarima['eta']) if modelo_sarima else '', 
                str(modelo_farima['arima_model']['eta']) if modelo_farima['arima_model'] else ''],
        'L_A': [str(modelo_sarima['L_A']) if modelo_sarima else '', 
                str(modelo_farima['arima_model']['L_A']) if modelo_farima['arima_model'] else ''],
        'L_M': [str(modelo_sarima['L_M']) if modelo_sarima else '', 
                str(modelo_farima['arima_model']['L_M']) if modelo_farima['arima_model'] else ''],
        'Gamma_hat_Phase1': [str(modelo_sarima['Gamma_hat']) if modelo_sarima else '',
                             str(modelo_farima['arima_model']['Gamma_hat']) if modelo_farima['arima_model'] else ''],
        'T_p': [np.nan, modelo_farima['T_p']],
        'K_p': [np.nan, modelo_farima['K_p']],
        'gamma': [np.nan, str(modelo_farima['gamma'])]
    }
    
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("train.csv", index=False)
    
    print("Entrenamiento finalizado. Resultados de configuración guardados en train.csv")
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
        return np.inf
    return np.log(sse) + (2 * (num_params + 2)) / denominador

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
        X[:, 2*(k-1)]     = np.cos(arg)
        X[:, 2*(k-1) + 1] = np.sin(arg)
    return X

# =============================================================================
# 2. ALGORITMOS SARIMA (TWO-PHASE OLS)
# =============================================================================

def sarima_fase_1(w, K_a):
    """
    Phase I: Estimación de Residuos vía AR(K_a) usando Pseudo-inversa SVD.
    """
    n_samples = len(w) - K_a
    if n_samples <= 0:
        return None, None

    Y = w[K_a:]                                 # w_{K_a}, w_{K_a+1}, ..., w_{T-1}
    Z = np.zeros((n_samples, K_a))

    for t in range(K_a, len(w)):
        # z_t = [w_{t-1}, w_{t-2}, ..., w_{t-K_a}]  ← ventana estrictamente pasada extraída con slicing seguro
        Z[t - K_a, :] = w[t - K_a : t][::-1]

    try:
        U, S_vals, VT = np.linalg.svd(Z, full_matrices=False)
        S_inv  = np.where(S_vals > 1e-10, 1.0 / S_vals, 0.0)
        Z_pinv = np.dot(VT.T, np.dot(np.diag(S_inv), U.T))
        Gamma_hat = np.dot(Z_pinv, Y)
    except np.linalg.LinAlgError:
        return None, None

    epsilon_hat = Y - np.dot(Z, Gamma_hat)
    return Gamma_hat, epsilon_hat

def construir_conjuntos_retardo(p, q, P, Q, s):
    """
    Construye los conjuntos de rezagos L_A y L_M.
    """
    s = int(s)

    L_A = set(range(1, p + 1))
    if s > 0:
        L_A.update(j * s       for j in range(1, P + 1))
        L_A.update(i + j * s   for i in range(1, p + 1)
                                for j in range(1, P + 1))
    L_A = {lag for lag in L_A if lag > 0}   # Filtro de seguridad

    L_M = set(range(1, q + 1))
    if s > 0:
        L_M.update(n * s       for n in range(1, Q + 1))
        L_M.update(m + n * s   for m in range(1, q + 1)
                                for n in range(1, Q + 1))
    L_M = {lag for lag in L_M if lag > 0}   # Filtro de seguridad

    return sorted(L_A), sorted(L_M)

def reconstruir_w_y_calcular_sse(w, epsilon_hat, K_a, L_A, L_M, eta_hat):
    """Evalúa la serie w_t ajustada con Fase II para obtener el SSE."""
    max_lag_A = max(L_A) if L_A else 0
    max_lag_M = max(L_M) if L_M else 0
    tau = max(K_a, max_lag_A, max_lag_M)

    if tau >= len(w):
        return np.inf, 0

    sse    = 0.0
    n_eval = 0
    for t in range(tau, len(w)):
        x_A_t = [w[t - l] for l in L_A]
        x_M_t = []
        for l in L_M:
            idx_eps = t - l - K_a
            if 0 <= idx_eps < len(epsilon_hat):
                x_M_t.append(epsilon_hat[idx_eps])
            else:
                x_M_t.append(0.0)

        X_t  = np.array(x_A_t + x_M_t)
        pred = np.dot(X_t, eta_hat)
        sse += (w[t] - pred) ** 2
        n_eval += 1

    return sse, n_eval

def grid_search_sarima(w, p_max, q_max, P_max, Q_max, s, K_a, lam):
    """Itera combinaciones minimizando AIC vía OLS."""
    Gamma_hat, epsilon_hat = sarima_fase_1(w, K_a)
    if Gamma_hat is None:
        return None

    best_aic    = np.inf
    best_params = None
    best_eta    = None
    best_L_A    = None
    best_L_M    = None

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            for P in range(P_max + 1):
                for Q in range(Q_max + 1):
                    if p == 0 and q == 0 and P == 0 and Q == 0:
                        continue

                    L_A, L_M = construir_conjuntos_retardo(p, q, P, Q, s)

                    # Si tras el filtro no quedaron rezagos, saltar
                    if not L_A and not L_M:
                        continue

                    max_lag_A = max(L_A) if L_A else 0
                    max_lag_M = max(L_M) if L_M else 0
                    tau = max(K_a, max_lag_A, max_lag_M)

                    if tau >= len(w):
                        continue

                    X_t_array = []
                    w_array   = []
                    for t in range(tau, len(w)):
                        x_A_t = [w[t - l] for l in L_A]
                        x_M_t = []
                        for l in L_M:
                            idx_eps = t - l - K_a
                            if 0 <= idx_eps < len(epsilon_hat):
                                x_M_t.append(epsilon_hat[idx_eps])
                            else:
                                x_M_t.append(0.0)
                        X_t_array.append(x_A_t + x_M_t)
                        w_array.append(w[t])

                    eta_hat = estimar_eta_sarima(X_t_array, w_array, lam)
                    if eta_hat is None:
                        continue

                    sse, n_eval = reconstruir_w_y_calcular_sse(
                        w, epsilon_hat, K_a, L_A, L_M, eta_hat
                    )

                    num_params = len(L_A) + len(L_M)
                    aic = calcular_aic(sse, n_eval, num_params)

                    if aic < best_aic:
                        best_aic    = aic
                        best_params = (p, q, P, Q)
                        best_eta    = eta_hat
                        best_L_A    = L_A
                        best_L_M    = L_M

    if best_params is None:
        return None

    return {
        'aic'      : float(best_aic),
        'order'    : best_params,
        'eta'      : best_eta.tolist(),
        'L_A'      : best_L_A,
        'L_M'      : best_L_M,
        'Gamma_hat': Gamma_hat.tolist() if Gamma_hat is not None else []
    }

# =============================================================================
# 3. ALGORITMOS F-ARIMA
# =============================================================================

def entrenar_farima(y, d, K_p_max, p_max, q_max, K_a, lam, s):
    """Entrena Fourier (Periodograma + OLS-Penalizado) y modela el residuo con ARIMA(p,d,q)"""

    t_n = np.arange(len(y))

    # Detrending lineal para evitar fuga espectral
    slope, intercept = np.polyfit(t_n, y, 1)
    y_detrended = y - (slope * t_n + intercept)

    f_k, I_fk = periodograma(y_detrended, f_s=1.0)

    # Excluir frecuencia 0 (componente DC) para buscar la frecuencia fundamental
    I_fk[0]  = 0
    idx_max  = np.argmax(I_fk)
    f_max    = f_k[idx_max]

    # T_p en número de muestras: T_p = 1 / f_max  (con f_s=1)
    T_p = (1.0 / f_max) if f_max > 0 else float(len(y))

    # Centrado de la serie
    y_mean    = np.mean(y)
    y_centered = y - y_mean

    best_aic_fourier = np.inf
    best_Kp          = 1
    best_gamma       = None
    best_residuals   = None

    for K_p in range(1, K_p_max + 1):
        X        = construir_matriz_fourier(t_n, T_p, K_p)
        gamma_hat = estimar_gamma_farima(X, y_centered, lam).flatten()
        pred     = np.dot(X, gamma_hat)
        res      = y_centered - pred

        # Residuos con la media restituida para preservar la escala de y
        res_con_media = res + y_mean

        sse        = np.sum(res ** 2)
        n_eval     = len(y)
        num_params = 2 * K_p
        aic        = calcular_aic(sse, n_eval, num_params)

        if aic < best_aic_fourier:
            best_aic_fourier = aic
            best_Kp          = K_p
            best_gamma       = gamma_hat
            best_residuals   = res_con_media

    # Componente ARIMA(p,d,q) sobre los residuos.
    w_residuals = diferenciar_serie(best_residuals, d, 0, 0)
    arima_res   = grid_search_sarima(w_residuals, p_max, q_max, 0, 0, 1, K_a, lam)

    return {
        'T_p'        : float(T_p),
        'K_p'        : best_Kp,
        'gamma'      : best_gamma.tolist() if best_gamma is not None else [],
        'arima_model': arima_res
    }

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Hiperparámetros
    train_size = 0.8
    p_max      = 2
    q_max      = 2
    P_max      = 1
    Q_max      = 1
    K_a        = 72
    lam        = 0.01
    K_p_max    = 5

    adf_results = pd.read_csv("adf.csv")
    d = int(adf_results['d_final'].iloc[-1])
    D = int(adf_results['D_final'].iloc[-1])
    s = int(adf_results['s'].iloc[-1])

    datos   = pd.read_csv("tserie.csv", header=None)
    y_full  = datos.iloc[:, 1].values

    n_train = int(len(y_full) * train_size)
    y_train = y_full[:n_train]

    # Entrenar SARIMA
    w_train       = diferenciar_serie(y_train, d, D, s)
    modelo_sarima = grid_search_sarima(w_train, p_max, q_max, P_max, Q_max, s, K_a, lam)

    # Entrenar F-ARIMA
    modelo_farima = entrenar_farima(y_train, d, K_p_max, p_max, q_max, K_a, lam, s)

    # Guardar train.csv
    resultados = {
        'modelo': ['SARIMA', 'FARIMA'],
        'order': [
            str(modelo_sarima['order'])                          if modelo_sarima                        else '',
            str(modelo_farima['arima_model']['order'])           if modelo_farima['arima_model']         else ''
        ],
        'eta': [
            str(modelo_sarima['eta'])                            if modelo_sarima                        else '',
            str(modelo_farima['arima_model']['eta'])             if modelo_farima['arima_model']         else ''
        ],
        'L_A': [
            str(modelo_sarima['L_A'])                            if modelo_sarima                        else '',
            str(modelo_farima['arima_model']['L_A'])             if modelo_farima['arima_model']         else ''
        ],
        'L_M': [
            str(modelo_sarima['L_M'])                            if modelo_sarima                        else '',
            str(modelo_farima['arima_model']['L_M'])             if modelo_farima['arima_model']         else ''
        ],
        'Gamma_hat_Phase1': [
            str(modelo_sarima['Gamma_hat'])                      if modelo_sarima                        else '',
            str(modelo_farima['arima_model']['Gamma_hat'])       if modelo_farima['arima_model']         else ''
        ],
        'T_p'  : [np.nan,              modelo_farima['T_p']],
        'K_p'  : [np.nan,              modelo_farima['K_p']],
        'gamma': [np.nan,  str(modelo_farima['gamma'])]
    }

    pd.DataFrame(resultados).to_csv("train.csv", index=False)
    print("Entrenamiento finalizado. Resultados de configuración guardados en train.csv")

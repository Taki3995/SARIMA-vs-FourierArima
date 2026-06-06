import numpy as np
import pandas as pd

# =============================================================================
# 1. FUNCIÓN DEL TEST ADF (Manual con numpy)
# =============================================================================

def ejecutar_test_adf(serie, max_lag, alpha):
    """
    Ejecuta el Augmented Dickey-Fuller Test de forma matricial con numpy.
    Modelo con constante: dy_t = c + gamma * y_{t-1} + sum(phi_i * dy_{t-i}) + e_t
    """
    y = np.array(serie)
    dy = np.diff(y)

    n_samples = len(y) - 1 - max_lag

    if n_samples <= 0:
        return 1.0, -2.86

    Y = dy[max_lag:]
    X = np.zeros((n_samples, 2 + max_lag))

    X[:, 0] = y[max_lag : -1]   # y_{t-1}
    X[:, 1] = 1.0               # Constante

    for i in range(max_lag):
        X[:, 2 + i] = dy[max_lag - 1 - i : -1 - i]

    # Aplicación de OLS explícito según requerimiento: (X^T X)^-1 X^T Y
    try:
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        beta = np.dot(XTX_inv, np.dot(X.T, Y))
    except np.linalg.LinAlgError:
        return 1.0, -2.86

    e = Y - np.dot(X, beta)
    df = n_samples - (2 + max_lag)

    if df <= 0:
        return 1.0, -2.86

    sigma2 = np.sum(e**2) / df
    cov_matrix = sigma2 * XTX_inv
    se_gamma = np.sqrt(cov_matrix[0, 0])

    if se_gamma == 0:
        return 1.0, -2.86

    t_stat = beta[0] / se_gamma

    cv = -3.43 if alpha <= 0.01 else (-2.86 if alpha <= 0.05 else -2.57)

    return t_stat, cv

# =============================================================================
# 2. FUNCIONES DE DIFERENCIACIÓN (Operadores L)
# =============================================================================

def diferenciar_ordinaria(y):
    """nabla^d y_t = y_t - y_{t-1}"""
    return np.diff(y, n=1)

def diferenciar_estacional(y, s):
    """nabla_s^D y_t = y_t - y_{t-s}"""
    return y[s:] - y[:-s]

# =============================================================================
# 3. LÓGICA DE INTEGRACIÓN (Cumplimiento de requerimientos)
# =============================================================================

def buscar_ordenes_integracion(serie, s, alpha=0.05, max_lag=30):
    historial = []
    y_current = np.array(serie)
    d, D = 0, 0

    # PASO 1: Determinar D (Diferenciación estacional) - Requiere usar nabla_s
    # El taller exige aplicar el operador de diferencia estacional
    y_seasonal = diferenciar_estacional(y_current, s)
    t_stat_D, cv_D = ejecutar_test_adf(y_seasonal, max_lag, alpha)
    
    if t_stat_D <= cv_D:
        D = 1
        y_current = y_seasonal
    
    historial.append({
        'etapa': 'estacional', 'orden': D,
        't_stat': round(t_stat_D, 4), 'valor_critico': cv_D,
        'es_estacionaria': bool(t_stat_D <= cv_D)
    })

    # PASO 2: Determinar d (Diferenciación ordinaria)
    t_stat_d, cv_d = ejecutar_test_adf(y_current, max_lag, alpha)
    if t_stat_d <= cv_d:
        d = 0
    else:
        d = 1
        y_current = diferenciar_ordinaria(y_current)

    historial.append({
        'etapa': 'ordinaria', 'orden': d,
        't_stat': round(t_stat_d, 4), 'valor_critico': cv_d,
        'es_estacionaria': bool(t_stat_d <= cv_d)
    })
    
    return d, D, historial

if __name__ == "__main__":
    # Configuración según requerimientos
    datos = pd.read_csv("tserie.csv", header=None)
    serie = datos.iloc[:, 1].values
    s = 24  # Valor típico para datos horarios según SARIMA.pdf

    d, D, historial = buscar_ordenes_integracion(serie, s)

    df_hist = pd.DataFrame(historial)
    df_hist.to_csv("adf.csv", index=False)
    print(f"Órdenes calculados: d={d}, D={D}")

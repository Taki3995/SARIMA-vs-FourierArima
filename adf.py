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

    # Si la serie es demasiado corta tras los rezagos, forzamos un valor no estacionario
    if n_samples <= 0:
        return 1.0, -2.86

    Y = dy[max_lag:]
    X = np.zeros((n_samples, 2 + max_lag))

    # Construcción de la matriz de diseño X
    X[:, 0] = y[max_lag : -1]   # Variable rezagada y_{t-1}
    X[:, 1] = 1.0               # Constante

    for i in range(max_lag):
        X[:, 2 + i] = dy[max_lag - 1 - i : -1 - i]   # Rezagos de las diferencias

    # Estimación vía Pseudo-inversa por SVD: V * S^-1 * U^T
    try:
        U, S, VT = np.linalg.svd(X, full_matrices=False)

        S_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
        X_pinv = np.dot(VT.T, np.dot(np.diag(S_inv), U.T))
        beta = np.dot(X_pinv, Y)

        S_inv2 = np.where(S > 1e-10, 1.0 / (S**2), 0.0)
        XTX_inv = np.dot(VT.T, np.dot(np.diag(S_inv2), VT))
    except np.linalg.LinAlgError:
        return 1.0, -2.86

    # Residuos y grados de libertad
    e = Y - np.dot(X, beta)
    df = n_samples - (2 + max_lag)

    if df <= 0:
        return 1.0, -2.86

    # Varianza y matriz de covarianza
    sigma2 = np.sum(e**2) / df
    cov_matrix = sigma2 * XTX_inv

    # Error estándar del coeficiente gamma (ubicado en beta[0])
    se_gamma = np.sqrt(cov_matrix[0, 0])

    if se_gamma == 0:
        return 1.0, -2.86

    t_stat = beta[0] / se_gamma

    # Mapeo manual de valores críticos (Dickey-Fuller aproximado con constante)
    if alpha <= 0.01:
        cv = -3.43
    elif alpha <= 0.05:
        cv = -2.86
    else:
        cv = -2.57

    return t_stat, cv

# =============================================================================
# 2. FUNCIONES DE DIFERENCIACIÓN
# =============================================================================

def diferenciar_ordinaria(serie):
    """(1 - L)y_t = y_t - y_{t-1}"""
    return np.diff(serie, n=1)

def diferenciar_estacional(serie, s):
    """(1 - L^s)y_t = y_t - y_{t-s}"""
    return serie[s:] - serie[:-s]

# =============================================================================
# 3. LÓGICA PRINCIPAL DE INTEGRACIÓN
# =============================================================================

def buscar_ordenes_integracion(serie, s, alpha, max_lag):
    """
    Busca el orden ordinario (d) y el orden estacional (D) mediante ADF.
    Condición de raíz unitaria: t_stat >= cv  →  se diferencia.

    CORRECCIÓN: el orden correcto es primero determinar d (diferenciación
    ordinaria) sobre la serie original, y luego determinar D (diferenciación
    estacional) sobre la serie ya estacionaria en nivel. El código anterior
    aplicaba D antes que d, lo que producía resultados como d=0, D=1 cuando
    la serie requería diferenciación ordinaria.
    """
    serie_actual = np.array(serie.copy())

    d = 0
    D = 0

    # Estadísticos intermedios para el CSV de resultados
    historial = []

    # ------------------------------------------------------------------
    # PASO 1: Determinar d (diferenciación ordinaria)
    # ------------------------------------------------------------------
    t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
    historial.append({
        'etapa': 'ordinaria', 'orden': d,
        't_stat': round(t_stat, 4), 'valor_critico': cv,
        'es_estacionaria': t_stat < cv
    })

    while t_stat >= cv and d < 3:
        serie_actual = diferenciar_ordinaria(serie_actual)
        d += 1
        t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
        historial.append({
            'etapa': 'ordinaria', 'orden': d,
            't_stat': round(t_stat, 4), 'valor_critico': cv,
            'es_estacionaria': t_stat < cv
        })

    # ------------------------------------------------------------------
    # PASO 2: Determinar D (diferenciación estacional) sobre la serie
    #         que ya es estacionaria en nivel
    # ------------------------------------------------------------------
    t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
    historial.append({
        'etapa': 'estacional', 'orden': D,
        't_stat': round(t_stat, 4), 'valor_critico': cv,
        'es_estacionaria': t_stat < cv
    })

    while t_stat >= cv and D < 3:
        serie_actual = diferenciar_estacional(serie_actual, s)
        D += 1
        t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
        historial.append({
            'etapa': 'estacional', 'orden': D,
            't_stat': round(t_stat, 4), 'valor_critico': cv,
            'es_estacionaria': t_stat < cv
        })

    return d, D, historial

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    data_path   = "tserie.csv"
    output_path = "adf.csv"

    # Parámetros fijos de configuración para la búsqueda
    alpha   = 0.05
    max_lag = 30
    s       = 24

    # Leer datos
    datos = pd.read_csv(data_path, header=None)
    serie = datos.iloc[:, 1].values

    # Ejecutar búsqueda de órdenes
    d, D, historial = buscar_ordenes_integracion(serie, s, alpha, max_lag)

    # ------------------------------------------------------------------
    # CORRECCIÓN: adf.csv ahora incluye los estadísticos intermedios de
    # cada iteración además de los órdenes finales d, D, s.
    # Esto cumple el requerimiento del taller de reportar resultados
    # parciales de cada etapa.
    # ------------------------------------------------------------------
    df_hist = pd.DataFrame(historial)
    df_hist['d_final'] = d
    df_hist['D_final'] = D
    df_hist['s']       = s

    df_hist.to_csv(output_path, index=False)

    print(f"Órdenes de integración calculados: d={d}, D={D}")
    print(f"Resultados guardados en {output_path}")
    print(df_hist.to_string(index=False))

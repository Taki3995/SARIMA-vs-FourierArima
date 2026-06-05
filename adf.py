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
    X[:, 0] = y[max_lag : -1]  # Variable rezagada y_{t-1}
    X[:, 1] = 1.0              # Constante
    
    for i in range(max_lag):
        X[:, 2 + i] = dy[max_lag - 1 - i : -1 - i] # Rezagos de las diferencias
        
    # Estimación OLS: beta = (X^T X)^-1 X^T Y
    try:
        XTX_inv = np.linalg.inv(np.dot(X.T, X))
    except np.linalg.LinAlgError:
        return 1.0, -2.86 # Falla por matriz singular
        
    beta = np.dot(np.dot(XTX_inv, X.T), Y)
    
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

# =============================================================================
# 3. LÓGICA PRINCIPAL DE INTEGRACIÓN
# =============================================================================

def buscar_orden_integracion(serie, alpha, max_lag):
    """
    Busca el orden ordinario (d) comparando el estadístico
    t_stat con el valor crítico cv. Condición de raíz unitaria: t_stat >= cv.
    """
    serie_actual = np.array(serie.copy())
    d = 0
    
    t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
    
    # Buscar orden ordinario (d) hasta un máximo lógico (ej. 3)
    while t_stat >= cv and d < 3:
        serie_actual = diferenciar_ordinaria(serie_actual)
        d += 1
        t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
        
    return d

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    data_path = "tserie.csv"
    output_path = "adf.csv"
    
    # Parámetros fijos de configuración para la búsqueda
    alpha = 0.05
    max_lag = 12
    
    # Leer datos (Corrección aplicada: header=None y selección de la columna 1)
    datos = pd.read_csv(data_path, header=None)
    serie = datos.iloc[:, 1].values 
    
    # Ejecutar búsqueda de orden
    d = buscar_orden_integracion(serie, alpha, max_lag)
    
    # Guardar resultados
    resultados = pd.DataFrame({
        'd': [d]
    })
    
    resultados.to_csv(output_path, index=False)
    print(f"Orden de integración calculado: d={d}")
    print(f"Resultados guardados en {output_path}")

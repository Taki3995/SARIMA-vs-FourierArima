import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. FUNCIONES MATEMÁTICAS AUXILIARES (Sin módulo math)
# =============================================================================

def factorial(n):
    """Calcula el factorial de un número entero."""
    if n == 0 or n == 1:
        return 1
    res = 1
    for i in range(2, int(n) + 1):
        res *= i
    return res

def combinatoria(n, k):
    """Calcula el coeficiente binomial (n sobre k)."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

# =============================================================================
# 2. FUNCIONES ARIMA (Two-Phase OLS)
# =============================================================================

def estimar_eta_arima(X_t_array, z_array):
    """
    Estima la representación lineal expandida (Fase II) vía OLS estándar.
    Fórmula: eta_hat = (sum X_t X_t^T)^-1 (sum X_t z_t)
    """
    if len(X_t_array) == 0:
        return None
        
    m = len(X_t_array[0]) # Dimensión del vector X_t
    
    sum_XX = np.zeros((m, m))
    sum_Xz = np.zeros((m, 1))
    
    # Sumatorias desde tau hasta T
    for t in range(len(X_t_array)):
        # Asegurar que X_t sea un vector columna
        X_t = np.array(X_t_array[t]).reshape(-1, 1)
        z_t = z_array[t]
        
        sum_XX += np.dot(X_t, X_t.T)
        sum_Xz += X_t * z_t
        
    try:
        mat_inv = np.linalg.inv(sum_XX)
    except np.linalg.LinAlgError:
        return None # Falla por matriz singular en OLS
        
    eta_hat = np.dot(mat_inv, sum_Xz)
    
    return eta_hat

def recuperar_arima(z_hat, Y_past, t_plus_h, d):
    """
    Recupera el valor original Y_{t+h} usando el Teorema del Binomio de Newton.
    Fórmula: Y_{t+h} = z_{t+h} - sum_{k=1}^{d} (-1)^{k} * comb(d,k) * Y_{t+h-k}
    """
    suma = 0.0
    for k in range(1, int(d) + 1):
        signo = (-1)**k
        coef_d = combinatoria(d, k)
        
        idx_pasado = t_plus_h - k
        
        # Extraer Y_{t+h-k}
        Y_val = Y_past.get(idx_pasado, 0.0) if isinstance(Y_past, dict) else Y_past[idx_pasado]
        
        suma += signo * coef_d * Y_val
            
    Y_t_h = z_hat - suma
    return Y_t_h

# =============================================================================
# 3. MÉTRICAS Y TESTS ESTADÍSTICOS
# =============================================================================

def calc_mnse(real, pred):
    """
    Calcula la Eficiencia de Nash-Sutcliffe Modificada (mNSE).
    Fórmula: mNSE = 1 - ( sum(|real - pred|) / sum(|real - mean(real)|) )
    """
    real = np.array(real)
    pred = np.array(pred)
    
    numerador = np.sum(np.abs(real - pred))
    denominador = np.sum(np.abs(real - np.mean(real)))
    
    # Evitar división por cero si la serie real es constante
    if denominador == 0:
        return np.nan
        
    mnse = 1.0 - (numerador / denominador)
    return mnse

def calc_mape(real, pred):
    """
    Calcula el Error Porcentual Absoluto Medio (MAPE).
    Fórmula: MAPE = (100 / N) * sum(|(real - pred) / real|)
    """
    real = np.array(real)
    pred = np.array(pred)
    
    # Máscara para evitar división por cero
    mask = real != 0
    real_safe = real[mask]
    pred_safe = pred[mask]
    
    n = len(real_safe)
    if n == 0:
        return np.nan
        
    mape = (100.0 / n) * np.sum(np.abs((real_safe - pred_safe) / real_safe))
    return mape

def test_jarque_bera(residuos):
    """
    Calcula el estadístico de Jarque-Bera.
    Fórmulas: 
    S (Asimetría) = m3 / m2^(3/2)
    K (Curtosis)  = m4 / m2^2
    JB = (N / 6) * (S^2 + ((K - 3)^2) / 4)
    """
    res = np.array(residuos)
    n = len(res)
    if n == 0:
        return np.nan
        
    media = np.mean(res)
    
    # Momentos centrales (usando el estimador poblacional dividiendo por n)
    m2 = np.sum((res - media)**2) / n
    m3 = np.sum((res - media)**3) / n
    m4 = np.sum((res - media)**4) / n
    
    # Evitar división por cero
    if m2 == 0:
        return np.nan
        
    S = m3 / (m2**(1.5))
    K = m4 / (m2**2)
    
    jb_stat = (n / 6.0) * (S**2 + ((K - 3.0)**2) / 4.0)
    
    return jb_stat

def calc_acf(residuos, lags):
    """
    Calcula la Función de Autocorrelación (ACF).
    Fórmula: r_k = sum((y_t - media)*(y_{t+k} - media)) / sum((y_t - media)^2)
    """
    res = np.array(residuos)
    n = len(res)
    media = np.mean(res)
    
    var_total = np.sum((res - media)**2)
    
    # Si la varianza es 0, no hay autocorrelación
    if var_total == 0:
        return np.zeros(lags + 1)
        
    acf_vals = []
    for k in range(lags + 1):
        if k == 0:
            acf_vals.append(1.0)
        else:
            # Producto de la serie truncada con su versión rezagada k pasos
            cov_k = np.sum((res[:n-k] - media) * (res[k:] - media))
            acf_vals.append(cov_k / var_total)
            
    return np.array(acf_vals)

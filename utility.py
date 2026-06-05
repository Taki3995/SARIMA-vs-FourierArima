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
# 2. FUNCIONES F-ARIMA
# =============================================================================

def periodograma(x):
    """
    Calcula el Periodograma de una serie de tiempo.
    Fórmula: I(f_k) = (1/N) * |X[k]|^2
    Bins de frecuencia positivos: K = floor(N/2)
    """
    N = len(x)
    # X[k] mediante Transformada de Fourier Discreta
    X_k = np.fft.fft(x)
    
    # Cálculo del periodograma I(f_k)
    I_fk = (1 / N) * (np.abs(X_k)**2)
    
    # Cálculo estricto de K para los bins positivos
    K = int(np.floor(N / 2))
    
    # Vector de frecuencias
    f_k = np.arange(K + 1) / N
    
    return f_k, I_fk[:K + 1]

def estimar_gamma_farima(X, Y, lam):
    """
    Estima los coeficientes de Fourier por OLS-Penalizado.
    Fórmula: Gamma_hat = (X^T X + lambda I)^-1 X^T Y
    """
    I = np.eye(X.shape[1])
    X_T = X.T
    
    mat_inv = np.linalg.inv(np.dot(X_T, X) + lam * I)
    gamma_hat = np.dot(np.dot(mat_inv, X_T), Y)
    
    return gamma_hat

# =============================================================================
# 3. FUNCIONES SARIMA
# =============================================================================

def estimar_eta_sarima(X_t_array, w_array, lam):
    """
    Estima la representación lineal expandida (Fase II) vía OLS Regularizado (Ridge).
    Fórmula: eta_hat = (sum X_t X_t^T + lambda I)^-1 (sum X_t w_t)
    """
    if len(X_t_array) == 0:
        return None
        
    m = len(X_t_array[0]) # Dimensión del vector X_t
    
    sum_XX = np.zeros((m, m))
    sum_Xw = np.zeros((m, 1))
    
    # Sumatorias desde tau hasta T
    for t in range(len(X_t_array)):
        # Asegurar que X_t sea un vector columna
        X_t = np.array(X_t_array[t]).reshape(-1, 1)
        w_t = w_array[t]
        
        sum_XX += np.dot(X_t, X_t.T)
        sum_Xw += X_t * w_t
        
    try:
        # Corrección: Añadir la regularización Ridge (lambda I) a la matriz de diseño
        I = np.eye(m)
        mat_inv = np.linalg.inv(sum_XX + lam * I)
    except np.linalg.LinAlgError:
        return None # Falla por matriz singular en OLS
        
    eta_hat = np.dot(mat_inv, sum_Xw)
    
    return eta_hat

def recuperar_sarima(w_t, y_past, t, d, D, s):
    """
    Recupera el valor original y_t usando el Teorema del Binomio de Newton.
    Fórmula: y_t = w_t - sum_{i=0..d, j=0..D, (i,j)!=(0,0)} (-1)^{i+j} * comb(d,i) * comb(D,j) * y_{t-i-js}
    """
    suma = 0.0
    for i in range(int(d) + 1):
        for j in range(int(D) + 1):
            if i == 0 and j == 0:
                continue # Se excluye la condición (0,0)
            
            signo = (-1)**(i + j)
            coef_d = combinatoria(d, i)
            coef_D = combinatoria(D, j)
            
            idx_pasado = t - i - j * s
            
            # Extraer y_{t-i-js}
            y_val = y_past.get(idx_pasado, 0.0) if isinstance(y_past, dict) else y_past[idx_pasado]
            
            suma += signo * coef_d * coef_D * y_val
            
    y_t = w_t - suma
    return y_t

# =============================================================================
# 4. MÉTRICAS Y TESTS ESTADÍSTICOS
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
    Fórmula: MAPE = mean(|(real - pred) / real|)
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
        
    # Corrección: Se elimina el multiplicador porcentual innecesario
    mape = (1.0 / n) * np.sum(np.abs((real_safe - pred_safe) / real_safe))
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
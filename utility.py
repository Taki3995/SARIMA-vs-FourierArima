import numpy as np
import pandas as pd
import math

def calculate_mape(y_true, y_pred):
    """
    Calcula el Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Se evita la división por cero agregando un epsilon o filtrando
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def calculate_mnse(y_true, y_pred):
    """
    Calcula el Modified Nash-Sutcliffe Efficiency (mNSE) usando diferencias absolutas.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_y_true = np.mean(y_true)
    
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - mean_y_true))
    
    if denominator == 0:
        return np.nan
        
    return 1 - (numerator / denominator)

def periodogram(series, fs=1):
    """
    Calcula el Periodograma para identificar componentes periódicas.
    
    Parámetros:
    series: array-like con la serie de tiempo temporal.
    fs: frecuencia de muestreo (por defecto 1 muestra/unidad de tiempo).
    """
    x = np.array(series)
    N = len(x)
    
    # Transformada de Fourier Discreta
    X_k = np.fft.fft(x)
    
    # Bins de Frecuencia positivos (K = floor(N/2))
    K = int(np.floor(N / 2))
    
    # Cálculo del Periodograma I(f_k) = (1/N) * |X[k]|^2
    I_fk = (1 / N) * (np.abs(X_k[:K + 1]) ** 2)
    
    # Vector de bins de frecuencia
    f_k = np.arange(K + 1) / N
    
    return f_k, I_fk

def ols_penalized(X, Y, lambda_param=0.0):
    """
    Estima los coeficientes expandidos vía método OLS-Penalizado (Ridge).
    Si lambda_param = 0, equivale a OLS ordinario de doble fase.
    """
    X = np.array(X)
    Y = np.array(Y)
    
    # Matriz Identidad para la regularización (penalización)
    I = np.eye(X.shape[1])
    
    # Estimación: (X^T X + lambda*I)^-1 X^T Y
    # Se utiliza pseudoinversa de Moore-Penrose para evitar problemas de singularidad si lambda=0
    inverse_term = np.linalg.pinv(X.T @ X + lambda_param * I)
    gamma_hat = inverse_term @ X.T @ Y
    
    return gamma_hat

def binomial_reconstruction(w_t, y_past, d, D, s, t_index):
    """
    Recupera la serie de tiempo a su dominio original usando el Teorema del Binomio de Newton.
    
    Parámetros:
    w_t: Valor diferenciado en el tiempo t.
    y_past: Array o Serie con los valores históricos originales de Y.
    d: Orden de diferenciación ordinaria.
    D: Orden de diferenciación estacional.
    s: Periodo estacional.
    t_index: Índice actual (t) referencial dentro de y_past.
    """
    y_t = w_t
    
    for i in range(d + 1):
        for j in range(D + 1):
            if i == 0 and j == 0:
                continue  # Se omite (i, j) = (0, 0) según la formulación
            
            # Coeficiente (-1)^(i+j) * (d sobre i) * (D sobre j)
            coef = ((-1)**(i + j)) * math.comb(d, i) * math.comb(D, j)
            
            # Cálculo del retardo
            lag = i + (j * s)
            
            # Recuperación: y_t = w_t - sum(...)
            y_t -= coef * y_past[t_index - lag]
            
    return y_t

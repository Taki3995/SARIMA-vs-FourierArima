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

def periodograma(x, f_s=1.0):
    """
    Calcula el Periodograma de una serie de tiempo.
    Fórmula: I(f_k) = (1/N) * |X[k]|^2, f_k = k * f_s / N
    Bins de frecuencia positivos: K = floor(N/2)
    """
    N = len(x)
    X_k = np.fft.fft(x)
    I_fk = (1 / N) * (np.abs(X_k)**2)
    K = int(np.floor(N / 2))
    f_k = (np.arange(K + 1) * f_s) / N
    return f_k, I_fk[:K + 1]

def estimar_gamma_farima(X, Y, lam):
    """
    Estima los coeficientes de Fourier por OLS-Penalizado.
    Fórmula: Gamma_hat = (X^T X + lambda I)^-1 X^T Y
    Usa np.linalg.solve en lugar de inv para mayor estabilidad numérica.
    """
    I = np.eye(X.shape[1])
    A = np.dot(X.T, X) + lam * I   # (X^T X + lambda I)
    b = np.dot(X.T, Y)              # X^T Y
    # CORRECCIÓN 1: solve(A, b) es más estable que inv(A) @ b
    gamma_hat = np.linalg.solve(A, b)
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

    m = len(X_t_array[0])

    sum_XX = np.zeros((m, m))
    sum_Xw = np.zeros(m)

    for t in range(len(X_t_array)):
        X_t = np.array(X_t_array[t])   # vector 1-D de dimensión m
        w_t = w_array[t]
        sum_XX += np.outer(X_t, X_t)   # X_t X_t^T
        sum_Xw += X_t * w_t            # X_t w_t

    try:
        I = np.eye(m)
        # CORRECCIÓN 1 (aplicada también aquí): solve en vez de inv
        eta_hat = np.linalg.solve(sum_XX + lam * I, sum_Xw)
    except np.linalg.LinAlgError:
        return None

    return eta_hat  # vector 1-D de dimensión m

def recuperar_sarima(w_t, y_past, t, d, D, s):
    """
    Recupera el valor original y_t usando el Teorema del Binomio de Newton.
    Fórmula: y_t = w_t - sum_{i=0..d, j=0..D, (i,j)!=(0,0)}
                         (-1)^{i+j} * C(d,i) * C(D,j) * y_{t-i-j*s}

    CORRECCIÓN 2: se verifica que idx_pasado sea un índice válido antes de
    acceder a y_past. Si el índice está fuera de rango la recuperación no es
    posible y se devuelve np.nan en lugar de silenciar el error con 0.0.
    """
    suma = 0.0
    for i in range(int(d) + 1):
        for j in range(int(D) + 1):
            if i == 0 and j == 0:
                continue

            signo   = (-1) ** (i + j)
            coef_d  = combinatoria(d, i)
            coef_D  = combinatoria(D, j)
            idx_pasado = t - i - j * int(s)

            # CORRECCIÓN 2: índice inválido → resultado indefinido
            if isinstance(y_past, dict):
                if idx_pasado not in y_past:
                    return np.nan
                y_val = y_past[idx_pasado]
            else:
                if idx_pasado < 0 or idx_pasado >= len(y_past):
                    return np.nan
                y_val = y_past[idx_pasado]

            suma += signo * coef_d * coef_D * y_val

    return w_t - suma

# =============================================================================
# 4. MÉTRICAS Y TESTS ESTADÍSTICOS
# =============================================================================

def calc_mnse(real, pred):
    """
    Calcula la Eficiencia de Nash-Sutcliffe Modificada (mNSE).
    Fórmula: mNSE = 1 - ( sum(|e_n|) / sum(|x_n - mean(x)|) )
    Sin cambios — implementación ya era correcta.
    """
    real = np.array(real)
    pred = np.array(pred)
    numerador   = np.sum(np.abs(real - pred))
    denominador = np.sum(np.abs(real - np.mean(real)))
    if denominador == 0:
        return np.nan
    return 1.0 - (numerador / denominador)

def calc_mape(real, pred):
    """
    Calcula el Error Porcentual Absoluto Medio (MAPE).
    Fórmula del taller: MAPE = mean(|e(n)/x(n)|)

    CORRECCIÓN 3: se devuelve el valor en fracción (0..1) tal como
    define la fórmula del taller. Si se desea expresar en porcentaje
    multiplicar el resultado por 100 al reportar, no aquí.
    El comentario anterior que decía "se elimina el multiplicador
    porcentual innecesario" era correcto matemáticamente pero causaba
    confusión. El valor 1.0 significa 100% de error.
    """
    real = np.array(real)
    pred = np.array(pred)
    mask = real != 0
    real_safe = real[mask]
    pred_safe = pred[mask]
    n = len(real_safe)
    if n == 0:
        return np.nan
    return (1.0 / n) * np.sum(np.abs((real_safe - pred_safe) / real_safe))

def test_jarque_bera(residuos):
    """
    Calcula el estadístico de Jarque-Bera y evalúa la normalidad.
    Fórmulas:
      S  = m3 / m2^(3/2)          (asimetría muestral)
      K  = m4 / m2^2               (curtosis muestral)
      JB = (N/6) * (S^2 + (K-3)^2/4)

    CORRECCIÓN 4: se devuelve un dict con el estadístico, el valor
    crítico chi²(2, alpha=0.05) = 5.991 y la conclusión, en lugar de
    solo el número. Esto cumple el objetivo del taller de "evaluar la
    significancia estadística".
    """
    res = np.array(residuos)
    n = len(res)
    if n == 0:
        return {'jb_stat': np.nan, 'critico_5pct': 5.991,
                'rechaza_H0': None, 'conclusion': 'Sin datos'}

    media = np.mean(res)
    m2 = np.sum((res - media) ** 2) / n
    m3 = np.sum((res - media) ** 3) / n
    m4 = np.sum((res - media) ** 4) / n

    if m2 == 0:
        return {'jb_stat': np.nan, 'critico_5pct': 5.991,
                'rechaza_H0': None, 'conclusion': 'Varianza cero'}

    S  = m3 / (m2 ** 1.5)
    K  = m4 / (m2 ** 2)
    jb = (n / 6.0) * (S ** 2 + ((K - 3.0) ** 2) / 4.0)

    critico = 5.991          # chi²(2, alpha=0.05)
    rechaza = bool(jb > critico)
    conclusion = ('Residuos NO normales (se rechaza H0)'
                  if rechaza else
                  'Residuos normales (no se rechaza H0)')

    return {
        'jb_stat'     : float(jb),
        'critico_5pct': critico,
        'rechaza_H0'  : rechaza,
        'conclusion'  : conclusion
    }

def calc_acf(residuos, lags):
    """
    Calcula la Función de Autocorrelación (ACF).
    Fórmula: r_k = cov_k / var_total
      donde cov_k = sum_{t=0}^{n-k-1} (y_t - media)(y_{t+k} - media)
            var_total = sum_{t=0}^{n-1}  (y_t - media)^2

    La implementación ya era matemáticamente correcta para la fórmula
    del taller. Sin cambios.
    """
    res   = np.array(residuos)
    n     = len(res)
    media = np.mean(res)
    var_total = np.sum((res - media) ** 2)

    if var_total == 0:
        return np.zeros(lags + 1)

    acf_vals = []
    for k in range(lags + 1):
        if k == 0:
            acf_vals.append(1.0)
        else:
            cov_k = np.sum((res[:n - k] - media) * (res[k:] - media))
            acf_vals.append(cov_k / var_total)

    return np.array(acf_vals)
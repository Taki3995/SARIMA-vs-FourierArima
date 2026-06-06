import numpy as np
import math

# =============================================================================
# 1. ESTIMADORES MATRICIALES (SVD y OLS)
# =============================================================================

def pinv_svd(A):
    """
    Calcula la matriz Pseudo-Inversa usando SVD para evitar singularidades.
    Fórmula: A^{-1} = V \times S^{-1} \times U^T
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    
    tol = np.max(S) * 1e-15
    S_inv = np.array([1/s if s > tol else 0 for s in S])
    
    invA = Vh.T @ np.diag(S_inv) @ U.T
    return invA

def ols_estimate(X, Y, lam=0.0):
    """
    Estimación OLS-Penalizado (Ridge) o Mínimos Cuadrados Ordinarios.
    Fórmula: \hat{\Gamma} = (X^T X + \lambda I)^{-1} X^T Y
    """
    I = np.eye(X.shape[1])
    A = (X.T @ X) + (lam * I)
    invA = pinv_svd(A)
    Gamma_hat = invA @ X.T @ Y
    return Gamma_hat

# =============================================================================
# 2. DIAGNÓSTICO ESTADÍSTICO (ADF, Jarque-Bera, ACF, Ruido Blanco)
# =============================================================================

def custom_adf(y, alpha=0.05):
    """
    Implementación manual del Test Dickey-Fuller Aumentado con rezago 1.
    Retorna: t_stat, crit_val, is_stationary
    """
    dy = np.diff(y)
    y_lag = y[:-1]
    
    dy_t = dy[1:]
    y_lag_1 = y_lag[1:]
    dy_lag = dy[:-1]
    
    N = len(dy_t)
    X = np.column_stack((np.ones(N), y_lag_1, dy_lag))
    
    Gamma = ols_estimate(X, dy_t)
    
    e = dy_t - (X @ Gamma)
    grados_libertad = N - X.shape[1]
    
    # Manejo de varianza cero para evitar divisiones inválidas
    sigma2 = np.sum(e**2) / grados_libertad
    if sigma2 == 0:
        return 0, -2.86, True

    var_Gamma = sigma2 * pinv_svd(X.T @ X)
    se_gamma = np.sqrt(np.abs(var_Gamma[1, 1])) 
    
    if se_gamma == 0:
        return 0, -2.86, True

    t_stat = Gamma[1] / se_gamma
    
    # Valores críticos aproximados Dickey-Fuller (sin tendencia, con constante)
    crit_vals = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}
    crit_val = crit_vals.get(alpha, -2.86)
    
    is_stationary = t_stat < crit_val
    return t_stat, crit_val, is_stationary

def jarque_bera(e):
    """
    Test de Jarque-Bera para evaluar normalidad de los residuos.
    Fórmulas: JB = (n/6) * (S^2 + 0.25*(K-3)^2)
    Retorna: jb_stat, is_normal
    """
    n = len(e)
    if n == 0: return 0, False
    
    mu = np.mean(e)
    sigma = np.std(e)
    
    if sigma == 0: return 0, False
    
    S = np.mean((e - mu)**3) / (sigma**3)
    K = np.mean((e - mu)**4) / (sigma**4)
    
    jb_stat = (n / 6.0) * (S**2 + 0.25 * (K - 3.0)**2)
    # Valor crítico de Chi-cuadrado con 2 grados de libertad al 95% es 5.991
    is_normal = jb_stat < 5.991
    return jb_stat, is_normal

def custom_acf(x, max_lag=None):
    n = len(x)
    if max_lag is None:
        max_lag = min(40, max(1, n // 4)) # Límite dinámico basado en la cantidad de datos
    
    mu = np.mean(x)
    var = np.var(x)
    if var == 0: return np.zeros(max_lag + 1)
        
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            if lag < n:
                cov = np.sum((x[lag:] - mu) * (x[:-lag] - mu)) / n
                acf[lag] = cov / var
    return acf

def es_ruido_blanco(e, max_lag=None):
    N = len(e)
    if N <= 1: return False
    
    if max_lag is None:
        max_lag = min(40, max(1, N // 4))
        
    limite = 1.96 / np.sqrt(N)
    lag_maximo = min(max_lag, N - 1)
    acf_vals = custom_acf(e, max_lag=lag_maximo)
    
    for lag in range(1, len(acf_vals)):
        if np.abs(acf_vals[lag]) > limite:
            return False 
    return True

# =============================================================================
# 3. MÉTTRICAS DE RENDIMIENTO Y CRITERIOS
# =============================================================================

def metricas_rendimiento(x_real, x_pred):
    """
    Retorna diccionario con MAE, RMSE, R2, MAPE, mNSE.
    """
    e = x_real - x_pred
    
    MAE = np.mean(np.abs(e))
    MSE = np.mean(e**2)
    RMSE = np.sqrt(MSE)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_array = np.abs(e / x_real)
        mape_array[~np.isfinite(mape_array)] = 0
        MAPE = np.mean(mape_array)
    
    var_e = np.var(e)
    var_y = np.var(x_real)
    R2 = 1 - (var_e / var_y) if var_y != 0 else 0
    
    mean_x = np.mean(x_real)
    numerador = np.sum(np.abs(e))
    denominador = np.sum(np.abs(x_real - mean_x))
    mNSE = 1 - (numerador / denominador) if denominador != 0 else 0
    
    return {'MAE': MAE, 'RMSE': RMSE, 'R2': R2, 'MAPE': MAPE, 'mNSE': mNSE}

def calcular_aic(e, p, N):
    """
    Criterio de Información Akaike.
    Fórmula: AIC = log(SSE) + 2(p+2) / (N-p-3)
    """
    SSE = np.sum(e**2)
    if SSE <= 0:
        SSE = 1e-10 # Prevenir log(0)
        
    denominador = (N - p - 3)
    if denominador <= 0:
        denominador = 1e-10 # Prevenir división por cero o negativa
        
    aic = np.log(SSE) + (2 * (p + 2)) / denominador
    return aic

# =============================================================================
# 4. TRANSFORMACIONES Y RECUPERACIÓN (Dominio de Frecuencia y Tiempo)
# =============================================================================

def periodograma(x, fs=1):
    """
    Calcula el Periodograma para detectar frecuencias dominantes.
    Solo retorna los Bins Positivos.
    """
    N = len(x)
    X_k = np.fft.fft(x)
    I_f = (1.0 / N) * (np.abs(X_k)**2)
    
    K = N // 2
    f = np.arange(K + 1) * (fs / N)
    I_f = I_f[:K + 1]
    
    return f, I_f

def nCr(n, r):
    """Coeficiente binomial (n sobre r)."""
    if r < 0 or r > n: return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n-r))

def recuperar_serie(w_t, y_hist, d, D, s):
    """
    Recupera el valor original usando el Teorema Binomial de Newton.
    Fórmula: y_t = w_t - \sum_{i,j \neq 0,0} (-1)^{i+j} (d i) (D j) y_{t-i-js}
    """
    suma = 0.0
    for i in range(d + 1):
        for j in range(D + 1):
            if i == 0 and j == 0:
                continue 
            
            coef = ((-1)**(i + j)) * nCr(d, i) * nCr(D, j)
            rezago = i + j * s
            
            if rezago <= len(y_hist):
                # Extraer el valor histórico contando hacia atrás
                suma += coef * y_hist[-rezago]
                
    y_t = w_t - suma
    return y_t
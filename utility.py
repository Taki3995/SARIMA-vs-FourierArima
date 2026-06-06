import numpy as np
import pandas as pd


def obtener_serie_objetivo(ruta_csv):
    datos = pd.read_csv(ruta_csv, header=None)
    datos = datos.apply(pd.to_numeric, errors='coerce')
    datos = datos.dropna(axis=0, how='any')
    datos = datos.sort_values(by=datos.columns[0])
    return datos.iloc[:, 1].to_numpy(dtype=float)


def factorial(n):
    if n == 0 or n == 1:
        return 1
    resultado = 1
    for i in range(2, int(n) + 1):
        resultado *= i
    return resultado


def combinatoria(n, k):
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


def periodograma(x, f_s=1.0):
    x = np.asarray(x, dtype=float)
    n_obs = len(x)
    espectro = np.fft.fft(x)
    intensidad = (1.0 / n_obs) * (np.abs(espectro) ** 2)
    k_max = int(np.floor(n_obs / 2))
    f_k = (np.arange(k_max + 1) * f_s) / n_obs
    return f_k, intensidad[: k_max + 1]


def estimar_gamma_farima(X, Y, lam):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sistema = X.T @ X + lam * np.eye(X.shape[1])
    termino = X.T @ Y
    return np.linalg.solve(sistema, termino)


def estimar_eta_sarima(X_t_array, w_array):
    if len(X_t_array) == 0:
        return None
    m = len(X_t_array[0])
    sum_XX = np.zeros((m, m))
    sum_Xw = np.zeros(m)
    for t in range(len(X_t_array)):
        X_t = np.array(X_t_array[t], dtype=float)
        w_t = w_array[t]
        sum_XX += np.outer(X_t, X_t)
        sum_Xw += X_t * w_t
    used_pinv = False
    try:
        eta_hat = np.linalg.solve(sum_XX, sum_Xw)
    except np.linalg.LinAlgError:
        eta_hat = np.dot(np.linalg.pinv(sum_XX), sum_Xw)
        used_pinv = True
    return eta_hat, used_pinv


def recuperar_sarima(w_t, y_past, t, d, D, s):
    suma = 0.0
    for i in range(int(d) + 1):
        for j in range(int(D) + 1):
            if i == 0 and j == 0:
                continue
            idx_pasado = t - i - j * int(s)
            if idx_pasado < 0 or idx_pasado >= len(y_past):
                return np.nan
            suma += ((-1) ** (i + j)) * combinatoria(d, i) * combinatoria(D, j) * y_past[idx_pasado]
    return w_t - suma


def calc_mnse(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    numerador = np.sum(np.abs(real - pred))
    denominador = np.sum(np.abs(real - np.mean(real)))
    if denominador == 0:
        return np.nan
    return 1.0 - (numerador / denominador)


def calc_mape(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mascara = real != 0
    real = real[mascara]
    pred = pred[mascara]
    if len(real) == 0:
        return np.nan
    return (1.0 / len(real)) * np.sum(np.abs((real - pred) / real))


def test_jarque_bera(residuos):
    res = np.asarray(residuos, dtype=float)
    res = res[~np.isnan(res)]
    n = len(res)
    if n == 0:
        return {
            'jb_stat': np.nan,
            'critico_5pct': 5.991,
            'rechaza_H0': None,
            'conclusion': 'Sin datos'
        }

    media = np.mean(res)
    m2 = np.sum((res - media) ** 2) / n
    if m2 == 0:
        return {
            'jb_stat': np.nan,
            'critico_5pct': 5.991,
            'rechaza_H0': None,
            'conclusion': 'Varianza cero'
        }

    m3 = np.sum((res - media) ** 3) / n
    m4 = np.sum((res - media) ** 4) / n
    s = m3 / (m2 ** 1.5)
    k = m4 / (m2 ** 2)
    jb = (n / 6.0) * (s ** 2 + ((k - 3.0) ** 2) / 4.0)
    critico = 5.991
    rechaza = bool(jb > critico)
    return {
        'jb_stat': float(jb),
        'critico_5pct': critico,
        'rechaza_H0': rechaza,
        'conclusion': 'Residuos NO normales (se rechaza H0)' if rechaza else 'Residuos normales (no se rechaza H0)'
    }


def calc_acf(residuos, lags):
    res = np.asarray(residuos, dtype=float)
    res = res[~np.isnan(res)]
    n = len(res)
    if n == 0:
        return np.full(lags + 1, np.nan)
    media = np.mean(res)
    var_total = np.sum((res - media) ** 2)
    if var_total == 0:
        acf = np.full(lags + 1, np.nan)
        acf[0] = 1.0
        return acf

    acf_vals = np.full(lags + 1, np.nan)
    acf_vals[0] = 1.0
    for k in range(1, min(lags, n - 1) + 1):
        cov_k = np.sum((res[: n - k] - media) * (res[k:] - media))
        acf_vals[k] = cov_k / var_total
    return acf_vals

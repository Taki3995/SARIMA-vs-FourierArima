import numpy as np
import pandas as pd

from utility import obtener_serie_objetivo


def ejecutar_test_adf(serie, max_lag, alpha):
    y = np.asarray(serie, dtype=float)
    dy = np.diff(y)
    n_samples = len(y) - 1 - max_lag

    if n_samples <= 0:
        return 1.0, -2.86

    Y = dy[max_lag:]
    X = np.zeros((n_samples, 2 + max_lag))
    X[:, 0] = y[max_lag:-1]
    X[:, 1] = 1.0

    for i in range(max_lag):
        X[:, 2 + i] = dy[max_lag - 1 - i : -1 - i]

    try:
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        S_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
        X_pinv = VT.T @ np.diag(S_inv) @ U.T
        beta = X_pinv @ Y
        S_inv2 = np.where(S > 1e-10, 1.0 / (S ** 2), 0.0)
        XTX_inv = VT.T @ np.diag(S_inv2) @ VT
    except np.linalg.LinAlgError:
        return 1.0, -2.86

    e = Y - X @ beta
    df = n_samples - (2 + max_lag)
    if df <= 0:
        return 1.0, -2.86

    sigma2 = np.sum(e ** 2) / df
    cov_matrix = sigma2 * XTX_inv
    se_gamma = np.sqrt(cov_matrix[0, 0])
    if se_gamma == 0:
        return 1.0, -2.86

    t_stat = beta[0] / se_gamma
    if alpha <= 0.01:
        cv = -3.43
    elif alpha <= 0.05:
        cv = -2.86
    else:
        cv = -2.57
    return t_stat, cv


def diferenciar_ordinaria(serie):
    return np.diff(serie, n=1)


def diferenciar_estacional(serie, s):
    return serie[s:] - serie[:-s]


def buscar_ordenes_integracion(serie, s, alpha, max_lag, d_max=3, D_max=3):
    serie_actual = np.asarray(serie, dtype=float)
    d = 0
    D = 0

    t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
    while t_stat >= cv and D < D_max:
        serie_actual = diferenciar_estacional(serie_actual, s)
        D += 1
        t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)

    t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)
    while t_stat >= cv and d < d_max:
        serie_actual = diferenciar_ordinaria(serie_actual)
        d += 1
        t_stat, cv = ejecutar_test_adf(serie_actual, max_lag, alpha)

    return d, D


if __name__ == '__main__':
    alpha = 0.05
    max_lag = 30
    s = 24
    serie = obtener_serie_objetivo('tserie.csv')
    d, D = buscar_ordenes_integracion(serie, s, alpha, max_lag)

    pd.DataFrame({
        'd': [d],
        'D': [D],
        's': [s],
        'alpha': [alpha],
        'max_lag': [max_lag]
    }).to_csv('adf.csv', index=False)

    print(f'd={d}, D={D}, s={s}')
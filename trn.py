import numpy as np
import pandas as pd

from utility import periodograma, estimar_gamma_farima, estimar_eta_sarima
from adf import buscar_ordenes_integracion


def calcular_aic(sse, n_obs, n_params):
    if sse <= 0 or n_obs <= 0:
        return np.inf
    return n_obs * np.log(sse / n_obs) + 2 * n_params


def diferenciar_serie(y, d, D, s):
    w = np.asarray(y, dtype=float)
    for _ in range(int(D)):
        w = w[s:] - w[:-s]
    for _ in range(int(d)):
        w = np.diff(w, n=1)
    return w


def construir_matriz_fourier(t_n, T_p_list, K_p):
    num_P = len(T_p_list)
    X = np.zeros((len(t_n), 2 * K_p * num_P))
    col = 0
    for T_p in T_p_list:
        for k in range(1, K_p + 1):
            arg = (2 * np.pi * k * t_n) / T_p
            X[:, col] = np.cos(arg)
            X[:, col + 1] = np.sin(arg)
            col += 2
    return X


def sarima_fase_1(w, K_a):
    w = np.asarray(w, dtype=float)
    n_samples = len(w) - K_a
    if n_samples <= 0:
        return None, None

    Y = w[K_a:]
    Z = np.zeros((n_samples, K_a))
    for t in range(n_samples):
        Z[t, :] = w[t : t + K_a][::-1]

    Gamma_hat, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    epsilon_hat = Y - Z @ Gamma_hat
    return Gamma_hat, epsilon_hat


def construir_conjuntos_retardo(p, q, P, Q, s):
    L_A = set(range(1, p + 1))
    L_A.update(j * s for j in range(1, P + 1))
    for i in range(1, p + 1):
        for j in range(1, P + 1):
            L_A.add(i + j * s)

    L_M = set(range(1, q + 1))
    L_M.update(n * s for n in range(1, Q + 1))
    for m in range(1, q + 1):
        for n in range(1, Q + 1):
            L_M.add(m + n * s)

    return sorted(L_A), sorted(L_M)


def orden_ar_optimo_fpe(w, max_order=None):
    """Encuentra el orden AR optimo via FPE sin techo arbitrario."""
    w = np.asarray(w, dtype=float)
    N = len(w)
    if max_order is None:
        max_order = int(np.ceil(np.log(N) * 10))

    best_fpe = np.inf
    best_p = 1

    for p in range(1, min(max_order, N // 4) + 1):
        Y = w[p:]
        Z = np.zeros((len(Y), p))
        for i in range(p):
            Z[:, i] = w[p - 1 - i : N - 1 - i]

        try:
            gamma, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
            e = Y - Z @ gamma
            sse = np.sum(e ** 2)
            n_eff = len(Y)
            fpe = (sse / n_eff) * ((n_eff + p) / (n_eff - p))
        except np.linalg.LinAlgError:
            continue

        if fpe < best_fpe:
            best_fpe = fpe
            best_p = p
        elif fpe > best_fpe * 1.01:
            break

    return best_p


def ajustar_sarima(w, p_max, q_max, P_max, Q_max, s, K_a):
    Gamma_hat, epsilon_hat = sarima_fase_1(w, K_a)
    if Gamma_hat is None:
        return None

    best_aic = np.inf
    best_result = None

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            for P in range(P_max + 1):
                for Q in range(Q_max + 1):
                    if p == 0 and q == 0 and P == 0 and Q == 0:
                        continue

                    L_A, L_M = construir_conjuntos_retardo(p, q, P, Q, s)
                    tau = max([K_a] + L_A + L_M)
                    if tau >= len(w):
                        continue

                    X_t_array = []
                    w_array = []
                    for t in range(tau, len(w)):
                        x_A_t = [w[t - l] for l in L_A]
                        x_M_t = []
                        for l in L_M:
                            idx_eps = t - l - K_a
                            if 0 <= idx_eps < len(epsilon_hat):
                                x_M_t.append(epsilon_hat[idx_eps])
                            else:
                                x_M_t.append(0.0)
                        X_t_array.append(x_A_t + x_M_t)
                        w_array.append(w[t])

                    res_eta = estimar_eta_sarima(X_t_array, w_array)
                    if res_eta is None:
                        continue
                    if isinstance(res_eta, tuple) or (hasattr(res_eta, '__len__') and len(res_eta) == 2):
                        eta_hat, _ = res_eta
                    else:
                        eta_hat = res_eta

                    sse = 0.0
                    n_eval = 0
                    for t in range(tau, len(w)):
                        x_A_t = [w[t - l] for l in L_A]
                        x_M_t = []
                        for l in L_M:
                            idx_eps = t - l - K_a
                            if 0 <= idx_eps < len(epsilon_hat):
                                x_M_t.append(epsilon_hat[idx_eps])
                            else:
                                x_M_t.append(0.0)
                        x_t = np.array(x_A_t + x_M_t, dtype=float)
                        pred = np.dot(x_t, eta_hat)
                        sse += (w[t] - pred) ** 2
                        n_eval += 1

                    aic = calcular_aic(sse, n_eval, len(L_A) + len(L_M))
                    if aic < best_aic:
                        best_aic = aic
                        best_result = {
                            'aic': float(aic),
                            'order': (p, q, P, Q),
                            'eta': eta_hat.tolist(),
                            'L_A': L_A,
                            'L_M': L_M,
                            'Gamma_hat': Gamma_hat.tolist()
                        }

    return best_result


def ajustar_farima(y, d, K_p_max, p_max, q_max, P_max, Q_max, K_a, lam, s, P_fourier=2):
    t_n = np.arange(len(y))

    y_estacionaria = diferenciar_serie(y, d, 0, 0)
    f_k, I_fk = periodograma(y_estacionaria, f_s=s)

    if len(I_fk) > 0:
        I_fk = I_fk.copy()
        I_fk[0] = 0

    T_p_list = []
    if len(I_fk) > 0:
        indices_top = np.argsort(I_fk)[-P_fourier:][::-1]
        for idx in indices_top:
            f_val = f_k[idx]
            if f_val > 0:
                T_p_list.append(np.round(s / f_val))

    if not T_p_list:
        T_p_list = [float(len(y))]

    best_aic = np.inf
    best_Kp = 1
    best_gamma = None
    best_residuals = None
    best_d_res = 0

    for K_p in range(1, K_p_max + 1):
        X = construir_matriz_fourier(t_n, T_p_list, K_p)
        gamma_hat = estimar_gamma_farima(X, y, lam).flatten()
        pred = X @ gamma_hat
        residuals = y - pred

        d_res, _ = buscar_ordenes_integracion(residuals, s=0, alpha=0.05, max_lag=30, d_max=3, D_max=0)
        w_residuals_aic = diferenciar_serie(residuals, d_res, 0, 0)

        aic = calcular_aic(np.sum(w_residuals_aic ** 2), len(w_residuals_aic), 2 * K_p * len(T_p_list))

        if aic < best_aic:
            best_aic = aic
            best_Kp = K_p
            best_gamma = gamma_hat
            best_residuals = residuals
            best_d_res = d_res

    w_residuals = diferenciar_serie(best_residuals, best_d_res, 0, 0)
    arima_model = ajustar_sarima(w_residuals, p_max, q_max, 0, 0, 0, K_a)

    return {
        'T_p': [float(tp) for tp in T_p_list],
        'K_p': best_Kp,
        'gamma': best_gamma.tolist() if best_gamma is not None else [],
        'arima_model': arima_model,
        'd_residual': best_d_res
    }


if __name__ == '__main__':
    train_size = 0.8
    lam = 0.01
    K_p_max = 5
    P_fourier = 2

    adf_results = pd.read_csv('adf.csv')
    d = int(adf_results['d'].iloc[0])
    D = int(adf_results['D'].iloc[0])
    s = int(adf_results['s'].iloc[0])

    datos = pd.read_csv('tserie.csv', header=None)
    datos = datos.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
    datos = datos.sort_values(by=datos.columns[0])
    y_full = datos.iloc[:, 1].to_numpy(dtype=float)

    n_train = int(len(y_full) * train_size)
    y_train = y_full[:n_train]

    # K_a automatico basado en la longitud del entrenamiento
    K_a = int(np.ceil(np.log(len(y_train)) * 10))

    w_train = diferenciar_serie(y_train, d, D, s)

    # Orden maximo adaptativo via FPE
    p_optimo = orden_ar_optimo_fpe(w_train)
    p_max = min(p_optimo + 1, len(w_train) // 10)
    q_max = p_max
    P_max = 1
    Q_max = 1

    print(f'K_a={K_a}, p_max={p_max}, q_max={q_max}, P_max={P_max}, Q_max={Q_max}')

    modelo_sarima = ajustar_sarima(w_train, p_max, q_max, P_max, Q_max, s, K_a)
    modelo_farima = ajustar_farima(y_train, d, K_p_max, p_max, q_max, P_max, Q_max, K_a, lam, s, P_fourier)

    resultados = [
        {
            'modelo': 'SARIMA',
            'order': str(modelo_sarima['order']) if modelo_sarima else '',
            'eta': str(modelo_sarima['eta']) if modelo_sarima else '',
            'L_A': str(modelo_sarima['L_A']) if modelo_sarima else '',
            'L_M': str(modelo_sarima['L_M']) if modelo_sarima else '',
            'Gamma_hat_Phase1': str(modelo_sarima['Gamma_hat']) if modelo_sarima else '',
            'T_p': '',
            'K_p': np.nan,
            'gamma': '',
            'aic': modelo_sarima['aic'] if modelo_sarima else np.nan,
            'train_size': train_size,
            'K_a': K_a,
            'p_max': p_max,
            'q_max': q_max,
            'P_max': P_max,
            'Q_max': Q_max,
            'K_p_max': K_p_max,
            'lam': lam,
            'P_fourier': np.nan,
            'd_residual': np.nan
        },
        {
            'modelo': 'FARIMA',
            'order': str(modelo_farima['arima_model']['order']) if modelo_farima and modelo_farima['arima_model'] else '',
            'eta': str(modelo_farima['arima_model']['eta']) if modelo_farima and modelo_farima['arima_model'] else '',
            'L_A': str(modelo_farima['arima_model']['L_A']) if modelo_farima and modelo_farima['arima_model'] else '',
            'L_M': str(modelo_farima['arima_model']['L_M']) if modelo_farima and modelo_farima['arima_model'] else '',
            'Gamma_hat_Phase1': str(modelo_farima['arima_model']['Gamma_hat']) if modelo_farima and modelo_farima['arima_model'] else '',
            'T_p': str(modelo_farima['T_p']) if modelo_farima else '',
            'K_p': modelo_farima['K_p'] if modelo_farima else np.nan,
            'gamma': str(modelo_farima['gamma']) if modelo_farima else '',
            'aic': modelo_farima['arima_model']['aic'] if modelo_farima and modelo_farima['arima_model'] else np.nan,
            'train_size': train_size,
            'K_a': K_a,
            'p_max': p_max,
            'q_max': q_max,
            'P_max': P_max,
            'Q_max': Q_max,
            'K_p_max': K_p_max,
            'lam': lam,
            'P_fourier': P_fourier,
            'd_residual': modelo_farima['d_residual'] if modelo_farima else np.nan
        }
    ]

    pd.DataFrame(resultados).to_csv('train.csv', index=False)
    print('Entrenamiento finalizado. Resultados guardados en train.csv')
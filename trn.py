import numpy as np
import pandas as pd

from utility import periodograma, estimar_gamma_farima, estimar_eta_sarima


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


def construir_matriz_fourier(t_n, T_p, K_p):
    X = np.zeros((len(t_n), 2 * K_p))
    for k in range(1, K_p + 1):
        arg = (2 * np.pi * k * t_n) / T_p
        X[:, 2 * (k - 1)] = np.cos(arg)
        X[:, 2 * (k - 1) + 1] = np.sin(arg)
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


def ajustar_farima(y, d, K_p_max, p_max, q_max, P_max, Q_max, K_a, lam, s):
    t_n = np.arange(len(y))
    y_estacionaria = diferenciar_serie(y, d, 0, 0)
    
    f_k, I_fk = periodograma(y_estacionaria, f_s=s)
    if len(I_fk) > 0:
        I_fk = I_fk.copy()
        I_fk[0] = 0
    idx_max = int(np.argmax(I_fk)) if len(I_fk) > 0 else 0
    f_max = f_k[idx_max] if len(f_k) > 0 else 0.0
    T_p = (s / f_max) if f_max > 0 else float(len(y))

    best_aic = np.inf
    best_Kp = 1
    best_gamma = None
    best_residuals = None

    for K_p in range(1, K_p_max + 1):
        X = construir_matriz_fourier(t_n, T_p, K_p)
        gamma_hat = estimar_gamma_farima(X, y, lam).flatten()
        pred = X @ gamma_hat
        residuals = y - pred
        w_residuals_aic = diferenciar_serie(residuals, d, 0, 0)
        
        aic = calcular_aic(np.sum(w_residuals_aic ** 2), len(w_residuals_aic), 2 * K_p)
        
        if aic < best_aic:
            best_aic = aic
            best_Kp = K_p
            best_gamma = gamma_hat
            best_residuals = residuals

    w_residuals = diferenciar_serie(best_residuals, d, 0, 0)
    arima_model = ajustar_sarima(w_residuals, p_max, q_max, 0, 0, 0, K_a)

    return {
        'T_p': float(T_p),
        'K_p': best_Kp,
        'gamma': best_gamma.tolist() if best_gamma is not None else [],
        'arima_model': arima_model
    }


if __name__ == '__main__':
    train_size = 0.8
    p_max = 2
    q_max = 2
    P_max = 1
    Q_max = 1
    K_a = 72
    lam = 0.01
    K_p_max = 5

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

    w_train = diferenciar_serie(y_train, d, D, s)
    modelo_sarima = ajustar_sarima(w_train, p_max, q_max, P_max, Q_max, s, K_a)
    modelo_farima = ajustar_farima(y_train, d, K_p_max, p_max, q_max, P_max, Q_max, K_a, lam, s)

    resultados = [
        {
            'modelo': 'SARIMA',
            'order': str(modelo_sarima['order']) if modelo_sarima else '',
            'eta': str(modelo_sarima['eta']) if modelo_sarima else '',
            'L_A': str(modelo_sarima['L_A']) if modelo_sarima else '',
            'L_M': str(modelo_sarima['L_M']) if modelo_sarima else '',
            'Gamma_hat_Phase1': str(modelo_sarima['Gamma_hat']) if modelo_sarima else '',
            'T_p': np.nan,
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
            'lam': lam
        },
        {
            'modelo': 'FARIMA',
            'order': str(modelo_farima['arima_model']['order']) if modelo_farima and modelo_farima['arima_model'] else '',
            'eta': str(modelo_farima['arima_model']['eta']) if modelo_farima and modelo_farima['arima_model'] else '',
            'L_A': str(modelo_farima['arima_model']['L_A']) if modelo_farima and modelo_farima['arima_model'] else '',
            'L_M': str(modelo_farima['arima_model']['L_M']) if modelo_farima and modelo_farima['arima_model'] else '',
            'Gamma_hat_Phase1': str(modelo_farima['arima_model']['Gamma_hat']) if modelo_farima and modelo_farima['arima_model'] else '',
            'T_p': modelo_farima['T_p'] if modelo_farima else np.nan,
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
            'lam': lam
        }
    ]

    pd.DataFrame(resultados).to_csv('train.csv', index=False)
    print('Entrenamiento finalizado. Resultados guardados en train.csv')
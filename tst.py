import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility import recuperar_sarima, calc_mnse, calc_mape, test_jarque_bera, calc_acf, obtener_serie_objetivo


def parse_list(s_val):
    if pd.isna(s_val):
        return []
    s = str(s_val).strip()
    if s in ('', '[]', 'None', 'nan'):
        return []
    s = s.replace('[', '').replace(']', '').replace('\n', '')
    valores = []
    for parte in s.split(','):
        parte = parte.strip()
        if parte == '':
            continue
        try:
            valores.append(float(parte))
        except ValueError:
            continue
    return valores


def diferenciar_serie_pad(y, d, D, s):
    w = np.asarray(y, dtype=float)
    for _ in range(int(D)):
        w_new = np.full_like(w, np.nan)
        w_new[s:] = w[s:] - w[:-s]
        w = w_new
    for _ in range(int(d)):
        w_new = np.full_like(w, np.nan)
        w_new[1:] = w[1:] - w[:-1]
        w = w_new
    return w


def calcular_residuos_empiricos(w, K_a, Gamma):
    epsilon = np.full_like(w, np.nan)
    if len(Gamma) == 0:
        return epsilon
    for t in range(int(K_a), len(w)):
        if not np.isnan(w[t - K_a : t + 1]).any():
            z_t = w[t - K_a : t][::-1]
            if len(z_t) == len(Gamma):
                epsilon[t] = w[t] - np.dot(z_t, Gamma)
    return epsilon


def construir_matriz_fourier(t_n, T_p_list, K_p):
    num_P = len(T_p_list)
    X = np.zeros((len(t_n), int(2 * K_p * num_P)))
    col = 0
    for T_p in T_p_list:
        for k in range(1, int(K_p) + 1):
            arg = (2 * np.pi * k * t_n) / T_p
            X[:, col] = np.cos(arg)
            X[:, col + 1] = np.sin(arg)
            col += 2
    return X


def predecir_arima_step(w_true, epsilon_true, t, L_A, L_M, eta):
    x_A_t = []
    for l in L_A:
        idx = t - int(l)
        x_A_t.append(w_true[idx] if idx >= 0 else np.nan)

    x_M_t = []
    for l in L_M:
        idx = t - int(l)
        x_M_t.append(epsilon_true[idx] if idx >= 0 else np.nan)

    x_t = np.array(x_A_t + x_M_t)
    if np.isnan(x_t).any():
        return np.nan
    return np.dot(x_t, eta)


if __name__ == '__main__':
    y_full = obtener_serie_objetivo('tserie.csv')

    adf_results = pd.read_csv('adf.csv')
    d = int(adf_results['d'].iloc[0])
    D = int(adf_results['D'].iloc[0])
    s = int(adf_results['s'].iloc[0])

    df_train = pd.read_csv('train.csv')
    row_sarima = df_train[df_train['modelo'] == 'SARIMA'].iloc[0]
    row_farima = df_train[df_train['modelo'] == 'FARIMA'].iloc[0]

    train_size = float(row_sarima['train_size']) if 'train_size' in df_train.columns and not pd.isna(row_sarima['train_size']) else 0.8
    K_a = int(float(row_sarima['K_a'])) if 'K_a' in df_train.columns and not pd.isna(row_sarima['K_a']) else 72

    start_index = int(len(y_full) * train_size)
    t_test = np.arange(start_index, len(y_full))
    y_true_test = y_full[start_index:]

    eta_sarima = np.array(parse_list(row_sarima['eta']))
    LA_sarima = [int(x) for x in parse_list(row_sarima['L_A'])]
    LM_sarima = [int(x) for x in parse_list(row_sarima['L_M'])]
    Gamma_sarima = np.array(parse_list(row_sarima['Gamma_hat_Phase1']))

    T_p_list = parse_list(row_farima['T_p'])
    K_p = int(float(row_farima['K_p']))
    gamma_fourier = np.array(parse_list(row_farima['gamma']))
    eta_farima = np.array(parse_list(row_farima['eta']))
    LA_farima = [int(x) for x in parse_list(row_farima['L_A'])]
    LM_farima = [int(x) for x in parse_list(row_farima['L_M'])]
    Gamma_farima = np.array(parse_list(row_farima['Gamma_hat_Phase1']))
    
    # Extraer d_residual específico para F-ARIMA si existe
    d_farima = int(float(row_farima['d_residual'])) if 'd_residual' in df_train.columns and not pd.isna(row_farima['d_residual']) else d

    w_true_sarima = diferenciar_serie_pad(y_full, d, D, s)
    eps_true_sarima = calcular_residuos_empiricos(w_true_sarima, K_a, Gamma_sarima)

    y_pred_sarima = []
    for t in t_test:
        w_hat_t = predecir_arima_step(w_true_sarima, eps_true_sarima, t, LA_sarima, LM_sarima, eta_sarima)
        if np.isnan(w_hat_t):
            y_pred_sarima.append(np.nan)
        else:
            y_pred_sarima.append(recuperar_sarima(w_hat_t, y_full, t, d, D, s))
    y_pred_sarima = np.array(y_pred_sarima)
    res_sarima = y_true_test - y_pred_sarima

    t_n_full = np.arange(len(y_full))
    X_fourier = construir_matriz_fourier(t_n_full, T_p_list, K_p)
    F_full = np.dot(X_fourier, gamma_fourier)
    residual_fourier = y_full - F_full

    # Aquí usamos el d_farima extraído
    w_true_farima = diferenciar_serie_pad(residual_fourier, d_farima, 0, 0)
    eps_true_farima = calcular_residuos_empiricos(w_true_farima, K_a, Gamma_farima)

    y_pred_farima = []
    for t in t_test:
        eta_hat_t = predecir_arima_step(w_true_farima, eps_true_farima, t, LA_farima, LM_farima, eta_farima)
        if np.isnan(eta_hat_t):
            y_pred_farima.append(np.nan)
        else:
            # Aquí también usamos el d_farima para la recuperación
            residual_hat_t = recuperar_sarima(eta_hat_t, residual_fourier, t, d_farima, 0, 0)
            y_pred_farima.append(F_full[t] + residual_hat_t)
    y_pred_farima = np.array(y_pred_farima)
    res_farima = y_true_test - y_pred_farima

    mask_s = ~np.isnan(y_pred_sarima)
    mask_f = ~np.isnan(y_pred_farima)
    jb_s = test_jarque_bera(res_sarima[mask_s])
    jb_f = test_jarque_bera(res_farima[mask_f])

    metrics = {
        'Modelo': ['SARIMA', 'FARIMA'],
        'mNSE': [calc_mnse(y_true_test[mask_s], y_pred_sarima[mask_s]), calc_mnse(y_true_test[mask_f], y_pred_farima[mask_f])],
        'MAPE': [calc_mape(y_true_test[mask_s], y_pred_sarima[mask_s]), calc_mape(y_true_test[mask_f], y_pred_farima[mask_f])],
        'JB_stat': [jb_s['jb_stat'], jb_f['jb_stat']],
        'JB_critico_5pct': [jb_s['critico_5pct'], jb_f['critico_5pct']],
        'JB_rechaza_H0': [jb_s['rechaza_H0'], jb_f['rechaza_H0']],
        'n_nan_pred': [int(np.isnan(y_pred_sarima).sum()), int(np.isnan(y_pred_farima).sum())],
        'n_valid_eval': [int(mask_s.sum()), int(mask_f.sum())]
    }
    pd.DataFrame(metrics).to_csv('test.csv', index=False)
    print('Evaluación finalizada. Métricas guardadas en test.csv')

    plt.figure(figsize=(14, 6))
    plt.plot(t_test, y_true_test, label='Valor Real', color='black', linewidth=2)
    plt.plot(t_test, y_pred_sarima, label='Predicción SARIMA', linestyle='--')
    plt.plot(t_test, y_pred_farima, label='Predicción F-ARIMA', linestyle='-.')
    plt.title('Pronóstico One-Step-Ahead: Valor Estimado vs Valor Real')
    plt.xlabel('Índice Temporal (Test)')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.savefig('predicciones.png')
    plt.close()

    lags = 20
    valid_s = res_sarima[mask_s]
    valid_f = res_farima[mask_f]
    acf_s = calc_acf(valid_s, min(lags, max(0, len(valid_s) - 1)))
    acf_f = calc_acf(valid_f, min(lags, max(0, len(valid_f) - 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    idx_s = np.arange(len(acf_s))
    idx_f = np.arange(len(acf_f))
    axes[0].bar(idx_s, acf_s, width=0.3, color='blue')
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].axhline(1.96 / np.sqrt(len(valid_s)), color='red', linestyle='--')
    axes[0].axhline(-1.96 / np.sqrt(len(valid_s)), color='red', linestyle='--')
    axes[0].set_title('ACF Residuos SARIMA')

    axes[1].bar(idx_f, acf_f, width=0.3, color='green')
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].axhline(1.96 / np.sqrt(len(valid_f)), color='red', linestyle='--')
    axes[1].axhline(-1.96 / np.sqrt(len(valid_f)), color='red', linestyle='--')
    axes[1].set_title('ACF Residuos F-ARIMA')

    plt.savefig('acf_residuos.png')
    plt.close()
    print('Gráficas guardadas: predicciones.png, acf_residuos.png')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import recuperar_sarima, calc_mnse, calc_mape, test_jarque_bera, calc_acf

# =============================================================================
# 1. FUNCIONES DE PARSEO (Sin librerías externas)
# =============================================================================

def parse_list(s_val):
    """Parsea un string de array a una lista de floats sin usar ast ni json."""
    if pd.isna(s_val): return []
    s_val = str(s_val).strip()
    if s_val in ('', '[]', 'None', 'nan'): return []
    s_val = s_val.replace('[', '').replace(']', '').replace('\n', '')
    return [float(x.strip()) for x in s_val.split(',') if x.strip() != '']

def parse_list_int(s_val):
    """
    Parsea un string de array a una lista de enteros.
    CORRECCIÓN: L_A y L_M se usan como índices; deben ser int para
    evitar errores de precisión de float (e.g. 24.0000001 → int=24 ✓,
    pero 23.9999999 → int=23 ✗ sin redondeo previo).
    """
    return [int(round(x)) for x in parse_list(s_val)]

# =============================================================================
# 2. FUNCIONES DE PROCESAMIENTO TEMPORAL
# =============================================================================

def diferenciar_serie_pad(y, d, D, s):
    """
    Diferencia la serie manteniendo la longitud original (rellenando con NaN).
    Permite alinear exactamente el índice temporal t entre y_t y w_t.
    """
    w = np.array(y, dtype=float)

    # Diferenciación estacional
    for _ in range(int(D)):
        w_new = np.full_like(w, np.nan)
        w_new[s:] = w[s:] - w[:-s]
        w = w_new

    # Diferenciación ordinaria
    for _ in range(int(d)):
        w_new = np.full_like(w, np.nan)
        w_new[1:] = w[1:] - w[:-1]
        w = w_new

    return w

def calcular_residuos_empiricos(w, K_a):
    # Aislar dinámicamente el inicio de los datos válidos tras diferenciaciones
    valid_indices = np.where(~np.isnan(w))[0]
    if len(valid_indices) == 0:
        raise ValueError("La serie diferenciada w contiene únicamente valores NaN.")
        
    start_idx = valid_indices[0]
    w_valid = w[start_idx:]
    T_valid = len(w_valid)
    
    if T_valid <= K_a:
        raise ValueError(f"Faltan grados de libertad. Longitud válida ({T_valid}) <= K_a ({K_a}).")
    
    # Construcción Vectorizada de la Matriz Z (Fase I)
    Z = []
    W_target = []
    for t in range(K_a, T_valid):
        Z.append(w_valid[t - K_a : t][::-1]) 
        W_target.append(w_valid[t])
        
    Z = np.array(Z)
    W_target = np.array(W_target)
    
    # Estimación OLS de innovaciones iniciales
    Gamma_hat = np.linalg.pinv(Z.T @ Z) @ Z.T @ W_target
    
    # Cálculo del residuo: epsilon_hat_t = w_t - z_t^T Gamma_hat
    epsilon_hat_valid = W_target - (Z @ Gamma_hat)
    
    # Reconstrucción del vector original mapeando los NaNs estructurales
    epsilon_hat = np.full(len(w), np.nan)
    epsilon_hat[start_idx + K_a : ] = epsilon_hat_valid
    
    return epsilon_hat, Gamma_hat

def construir_matriz_fourier(t_n, T_p, K_p):
    """Construye la matriz de diseño de Fourier para toda la serie temporal."""
    n_samples = len(t_n)
    X = np.zeros((n_samples, int(2 * K_p)))
    for k in range(1, int(K_p) + 1):
        arg = (2 * np.pi * k * t_n) / T_p
        X[:, 2*(k-1)]     = np.cos(arg)
        X[:, 2*(k-1) + 1] = np.sin(arg)
    return X

# =============================================================================
# 3. LÓGICA DE PREDICCIÓN ONE-STEP-AHEAD
# =============================================================================

def predecir_arima_step(w_true, epsilon_true, t, L_A, L_M, eta):
    """
    Calcula la predicción w_hat_t usando los retardos pasados verdaderos.
    """
    x_A_t = []
    for l in L_A:
        idx = t - int(l)
        x_A_t.append(w_true[idx] if idx >= 0 else np.nan)

    x_M_t = []
    for l in L_M:
        idx = t - int(l)
        x_M_t.append(epsilon_true[idx] if idx >= 0 else np.nan)

    X_t = np.array(x_A_t + x_M_t)

    if np.isnan(X_t).any():
        return np.nan

    return np.dot(X_t, eta)

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # 1. Cargar serie de tiempo
    datos  = pd.read_csv("tserie.csv", header=None)
    y_full = datos.iloc[:, 1].values

    train_size  = 0.8
    K_a         = 72   # Debe coincidir con el usado en trn.py

    start_index = int(len(y_full) * train_size)
    end_index   = len(y_full)

    t_test      = np.arange(start_index, end_index)
    y_true_test = y_full[start_index:end_index]

    # 2. Cargar resultados de etapas anteriores
    # CORRECCIÓN: adf.csv ahora tiene múltiples filas (una por iteración ADF).
    # Se leen d_final, D_final, s de la última fila, donde ya convergieron.
    adf_results = pd.read_csv("adf.csv")
    d = int(adf_results['d_final'].iloc[-1])
    D = int(adf_results['D_final'].iloc[-1])
    s = int(adf_results['s'].iloc[-1])

    df_train = pd.read_csv("train.csv")

    # 3. Extraer parámetros SARIMA
    row_sarima   = df_train[df_train['modelo'] == 'SARIMA'].iloc[0]
    eta_sarima   = np.array(parse_list(row_sarima['eta']))
    # CORRECCIÓN: L_A y L_M se parsean como enteros para uso como índices
    LA_sarima    = parse_list_int(row_sarima['L_A'])
    LM_sarima    = parse_list_int(row_sarima['L_M'])
    Gamma_sarima = np.array(parse_list(row_sarima['Gamma_hat_Phase1']))

    # 4. Extraer parámetros F-ARIMA
    row_farima    = df_train[df_train['modelo'] == 'FARIMA'].iloc[0]
    T_p           = float(row_farima['T_p'])
    K_p           = int(float(row_farima['K_p']))
    gamma_fourier = np.array(parse_list(row_farima['gamma']))
    eta_farima    = np.array(parse_list(row_farima['eta']))
    # CORRECCIÓN: L_A y L_M se parsean como enteros para uso como índices
    LA_farima     = parse_list_int(row_farima['L_A'])
    LM_farima     = parse_list_int(row_farima['L_M'])
    Gamma_farima  = np.array(parse_list(row_farima['Gamma_hat_Phase1']))

    # ---------------------------------------------------------
    # 5. PREDICCIÓN SARIMA (One-Step-Ahead)
    # ---------------------------------------------------------
    w_true_sarima   = diferenciar_serie_pad(y_full, d, D, s)
    eps_true_sarima = calcular_residuos_empiricos(w_true_sarima, K_a, Gamma_sarima)

    y_pred_sarima = []

    for t in t_test:
        w_hat_t = predecir_arima_step(
            w_true_sarima, eps_true_sarima, t, LA_sarima, LM_sarima, eta_sarima
        )
        if np.isnan(w_hat_t):
            y_pred_sarima.append(np.nan)
        else:
            y_hat_t = recuperar_sarima(w_hat_t, y_full, t, d, D, s)
            y_pred_sarima.append(y_hat_t)

    y_pred_sarima = np.array(y_pred_sarima)
    res_sarima    = y_true_test - y_pred_sarima

    # ---------------------------------------------------------
    # 6. PREDICCIÓN F-ARIMA (One-Step-Ahead)
    # ---------------------------------------------------------
    t_n_full  = np.arange(len(y_full))
    X_fourier = construir_matriz_fourier(t_n_full, T_p, K_p)
    F_full    = np.dot(X_fourier, gamma_fourier)

    residual_fourier = y_full - F_full

    w_true_farima   = diferenciar_serie_pad(residual_fourier, d, 0, 0)
    eps_true_farima = calcular_residuos_empiricos(w_true_farima, K_a, Gamma_farima)

    y_pred_farima = []

    for t in t_test:
        eta_hat_t = predecir_arima_step(
            w_true_farima, eps_true_farima, t, LA_farima, LM_farima, eta_farima
        )
        if np.isnan(eta_hat_t):
            y_pred_farima.append(np.nan)
        else:
            residual_hat_t = recuperar_sarima(eta_hat_t, residual_fourier, t, d, 0, 0)
            # Predicción final = componente Fourier + residuo ARIMA recuperado
            y_hat_t = F_full[t] + residual_hat_t
            y_pred_farima.append(y_hat_t)

    y_pred_farima = np.array(y_pred_farima)
    res_farima    = y_true_test - y_pred_farima

    # ---------------------------------------------------------
    # 7. EVALUACIÓN Y MÉTRICAS
    # ---------------------------------------------------------
    mask_s = ~np.isnan(y_pred_sarima)
    mask_f = ~np.isnan(y_pred_farima)

    # test_jarque_bera ahora devuelve un dict; se desempaca cada campo
    jb_sarima = test_jarque_bera(res_sarima[mask_s])
    jb_farima = test_jarque_bera(res_farima[mask_f])

    # CORRECCIÓN: calc_mape devuelve fracción (0..1); se multiplica por 100
    # al reportar para expresar el MAPE en porcentaje, como es convención.
    metrics = {
        'Modelo'          : ['SARIMA', 'FARIMA'],
        'mNSE'            : [
            calc_mnse(y_true_test[mask_s], y_pred_sarima[mask_s]),
            calc_mnse(y_true_test[mask_f], y_pred_farima[mask_f])
        ],
        'MAPE_pct'        : [
            calc_mape(y_true_test[mask_s], y_pred_sarima[mask_s]) * 100,
            calc_mape(y_true_test[mask_f], y_pred_farima[mask_f]) * 100
        ],
        # CORRECCIÓN: columnas separadas del resultado Jarque-Bera
        'JB_stat'         : [jb_sarima['jb_stat'],      jb_farima['jb_stat']],
        'JB_critico_5pct' : [jb_sarima['critico_5pct'], jb_farima['critico_5pct']],
        'JB_rechaza_H0'   : [jb_sarima['rechaza_H0'],   jb_farima['rechaza_H0']],
        'JB_conclusion'   : [jb_sarima['conclusion'],   jb_farima['conclusion']]
    }

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("test.csv", index=False)
    print("Evaluación finalizada. Métricas guardadas en test.csv")
    print(df_metrics.to_string(index=False))

    # CORRECCIÓN: guardar predicciones por índice temporal para reproducibilidad
    pd.DataFrame({
        't'             : t_test,
        'y_real'        : y_true_test,
        'y_pred_sarima' : y_pred_sarima,
        'y_pred_farima' : y_pred_farima,
        'res_sarima'    : res_sarima,
        'res_farima'    : res_farima
    }).to_csv("predicciones.csv", index=False)

    # ---------------------------------------------------------
    # 8. GRÁFICAS DE RESULTADOS
    # ---------------------------------------------------------

    # Gráfica 1: Curvas Comparativas
    plt.figure(figsize=(14, 6))
    plt.plot(t_test, y_true_test,  label='Valor Real',         color='black', linewidth=2)
    plt.plot(t_test, y_pred_sarima, label='Predicción SARIMA', linestyle='--')
    plt.plot(t_test, y_pred_farima, label='Predicción F-ARIMA', linestyle='-.')
    plt.title("Pronóstico One-Step-Ahead: Valor Estimado vs Valor Real")
    plt.xlabel("Índice Temporal (Test)")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predicciones.png")
    plt.show()

    # Gráfica 2: ACF de residuos
    lags        = 20
    acf_sarima  = calc_acf(res_sarima[mask_s], lags)
    acf_farima  = calc_acf(res_farima[mask_f],  lags)
    lag_indices = np.arange(lags + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(lag_indices, acf_sarima, width=0.3, color='blue')
    axes[0].axhline(0,  color='black', linewidth=1)
    axes[0].axhline( 1.96 / np.sqrt(len(res_sarima[mask_s])), color='red', linestyle='--')
    axes[0].axhline(-1.96 / np.sqrt(len(res_sarima[mask_s])), color='red', linestyle='--')
    axes[0].set_title("ACF Residuos SARIMA")

    axes[1].bar(lag_indices, acf_farima, width=0.3, color='green')
    axes[1].axhline(0,  color='black', linewidth=1)
    axes[1].axhline( 1.96 / np.sqrt(len(res_farima[mask_f])), color='red', linestyle='--')
    axes[1].axhline(-1.96 / np.sqrt(len(res_farima[mask_f])), color='red', linestyle='--')
    axes[1].set_title("ACF Residuos F-ARIMA")

    plt.tight_layout()
    plt.savefig("acf_residuos.png")
    plt.show()

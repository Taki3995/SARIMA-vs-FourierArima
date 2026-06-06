import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importamos las herramientas de nuestro core matemático
from utility import metricas_rendimiento, jarque_bera, es_ruido_blanco, recuperar_serie, ols_estimate

def fase_I_innovaciones_dinamicas(w, K_a):
    """Reconstruye las innovaciones para toda la serie usando el K_a óptimo."""
    eps = np.zeros(len(w))
    if K_a >= len(w): return eps
    
    Y_ar = w[K_a:]
    X_ar = np.array([w[i-K_a : i][::-1] for i in range(K_a, len(w))])
    Gamma_ar = ols_estimate(X_ar, Y_ar, lam=0.0)
    eps[K_a:] = Y_ar - (X_ar @ Gamma_ar)
    return eps

def main():
    print("-> Iniciando tst.py (Evaluación y Pronóstico One-Step-Ahead)...")
    
    # 1. Cargar Datos y Archivos de Configuración
    try:
        df_y = pd.read_csv('tserie.csv')
        y = df_y.iloc[:, 1].values if df_y.shape[1] > 1 else df_y.iloc[:, 0].values
        
        df_adf = pd.read_csv('adf.csv')
        d, D, s = int(df_adf['d'].iloc[0]), int(df_adf['D'].iloc[0]), int(df_adf['s'].iloc[0])
        
        df_farima = pd.read_csv('train_farima.csv')
        freqs = df_farima['frecuencias'].dropna().values
        gamma_f = df_farima['coeficientes'].values
        
        df_sarima = pd.read_csv('train_sarima.csv')
        p, q, P, Q = int(df_sarima['p'].iloc[0]), int(df_sarima['q'].iloc[0]), int(df_sarima['P'].iloc[0]), int(df_sarima['Q'].iloc[0])
        K_a = int(df_sarima['K_a'].iloc[0])
        eta = df_sarima.filter(like='coef_').values[0]
        
    except FileNotFoundError as e:
        print(f"[Error] No se encontró un archivo generado previamente: {e}")
        return

    # 2. Definición Dinámica del Conjunto de Prueba (Test Set = Último 20%)
    N = len(y)
    test_size = int(N * 0.20)
    train_size = N - test_size
    
    # Índices del Test Set
    t_test = np.arange(train_size, N)
    y_test_real = y[train_size:]
    
    print(f"   -> Total de datos: {N} | Entrenados: {train_size} | A pronosticar (Test): {test_size}")

    # =========================================================
    # PRONÓSTICO F-ARIMA (One-Step-Ahead)
    # =========================================================
    print("\n   -> Ejecutando pronóstico F-ARIMA...")
    X_f = [np.ones(test_size)]
    for freq in freqs:
        X_f.append(np.cos(2 * np.pi * freq * t_test))
        X_f.append(np.sin(2 * np.pi * freq * t_test))
    X_f = np.column_stack(X_f)
    y_pred_farima = X_f @ gamma_f

    # =========================================================
    # PRONÓSTICO SARIMA (One-Step-Ahead)
    # =========================================================
    print("   -> Ejecutando pronóstico SARIMA...")
    # Diferenciar la serie original completa para tener todo el historial w_t
    w = y.copy()
    for _ in range(D): w = w[s:] - w[:-s]
    for _ in range(d): w = w[1:] - w[:-1]
    
    # Recalcular las innovaciones base para la serie diferenciada
    eps = fase_I_innovaciones_dinamicas(w, K_a)
    
    y_pred_sarima = []
    offset = d + D * s # Desplazamiento de índices por la pérdida de datos al diferenciar
    
    for t in t_test:
        k = t - offset # Índice equivalente en el dominio w_t
        
        # Construir matriz de diseño x_t para el paso actual
        fila = [1.0] # Intercepto
        for i in range(1, p + 1): fila.append(w[k - i])
        for j in range(1, P + 1): fila.append(w[k - j * s])
        for i in range(1, q + 1): fila.append(eps[k - i])
        for j in range(1, Q + 1): fila.append(eps[k - j * s])
        
        # Pronóstico en dominio diferenciado
        w_hat = np.dot(fila, eta)
        
        # Recuperación al dominio original mediante Newton (le pasamos la historia hasta t-1)
        y_hat = recuperar_serie(w_hat, y[:t], d, D, s)
        y_pred_sarima.append(y_hat)
        
    y_pred_sarima = np.array(y_pred_sarima)

    # =========================================================
    # EVALUACIÓN Y MÉTRICAS FINALES
    # =========================================================
    print("\n--- REPORTE DE RESULTADOS ---")
    
    # 1. Métricas F-ARIMA
    e_farima = y_test_real - y_pred_farima
    met_f = metricas_rendimiento(y_test_real, y_pred_farima)
    jb_stat_f, is_norm_f = jarque_bera(e_farima)
    rb_f = es_ruido_blanco(e_farima)
    
    print("\n[ F-ARIMA ]")
    print(f"  mNSE:  {met_f['mNSE']:.4f}")
    print(f"  MAPE:  {met_f['MAPE']:.4f}")
    print(f"  Residuos -> Normalidad (JB): {'Pasa' if is_norm_f else 'Falla'} | Ruido Blanco (ACF): {'Sí' if rb_f else 'No'}")

    # 2. Métricas SARIMA
    e_sarima = y_test_real - y_pred_sarima
    met_s = metricas_rendimiento(y_test_real, y_pred_sarima)
    jb_stat_s, is_norm_s = jarque_bera(e_sarima)
    rb_s = es_ruido_blanco(e_sarima)
    
    print("\n[ SARIMA ]")
    print(f"  mNSE:  {met_s['mNSE']:.4f}")
    print(f"  MAPE:  {met_s['MAPE']:.4f}")
    print(f"  Residuos -> Normalidad (JB): {'Pasa' if is_norm_s else 'Falla'} | Ruido Blanco (ACF): {'Sí' if rb_s else 'No'}")

    # 3. Exportar resultados a CSV
    df_test_results = pd.DataFrame({
        'Modelo': ['F-ARIMA', 'SARIMA'],
        'mNSE': [met_f['mNSE'], met_s['mNSE']],
        'MAPE': [met_f['MAPE'], met_s['MAPE']],
        'Test_Size': [test_size, test_size]
    })
    df_test_results.to_csv('test.csv', index=False)
    print("\n-> [OK] Métricas finales exportadas a 'test.csv'.")

    # =========================================================
    # VISUALIZACIÓN
    # =========================================================
    print("-> Generando gráficos comparativos...")
    plt.figure(figsize=(12, 6))
    
    # Mostramos un poco de contexto (últimos 50 datos de entrenamiento + el Test Set)
    contexto = 50
    t_context = np.arange(train_size - contexto, train_size)
    
    plt.plot(t_context, y[train_size - contexto:train_size], color='gray', label='Historia (Entrenamiento)', linestyle='--')
    plt.plot(t_test, y_test_real, color='black', linewidth=2, label='Valor Real (Test)')
    plt.plot(t_test, y_pred_farima, color='blue', label='Predicción F-ARIMA', alpha=0.7)
    plt.plot(t_test, y_pred_sarima, color='red', label='Predicción SARIMA', alpha=0.7)
    
    plt.axvline(x=train_size, color='green', linestyle=':', label='Inicio One-Step-Ahead')
    
    plt.title('Comparativa de Pronóstico One-Step-Ahead (F-ARIMA vs SARIMA)')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico y mostrarlo
    plt.savefig('grafico_comparativo.png', dpi=300)
    print("-> [OK] Gráfico guardado como 'grafico_comparativo.png'.")
    plt.show()

if __name__ == "__main__":
    main()
import functions
import data
import visualizations

# Datos de entrenamiento y Prueba
train_df = data.data_open_2("AUDUSD_train.csv")
test_df = data.data_open_2("AUDUSD_test.csv")

# Preprocesamiento de dataframes para evaluación de estrategia
# Creación de indicadores
# Exponential Moving Average y Aroon Oscillator
train_df = (
    train_df.pipe(functions.calculate_ema, 20)
            .pipe(functions.calculate_aroon_oscillator)
)
# Parabolic SAR
train_df["Parabolic_SAR"] = functions.calculate_parabolic_sar(train_df)
# Exponential Moving Average y Aroon Oscillator
test_df = (
    test_df.pipe(functions.calculate_ema, 20)
            .pipe(functions.calculate_aroon_oscillator)
)
# Parabolic SAR
test_df["Parabolic_SAR"] = functions.calculate_parabolic_sar(test_df)

# Optimización de parámetros y Backtesting (sólo sobre Train se realiza)
best_params_train,iter_val_test = functions.best_fit_params(train_df)
print("Best take_profit_ratio:", best_params_train[0])
print("Best stop_loss_ratio:", best_params_train[1])
print("Best position_size:", best_params_train[2])

# Creación del portafolio a partir de la estrategia con parámetros optimizados
# Backtesting sobre entrenamiento
portfolio = functions.run_trading_strategy(train_df,
                                             capital=100000,
                                             take_profit_ratio=best_params_train[0],
                                             stop_loss_ratio=best_params_train[1],
                                             position_size=best_params_train[2])
# Sobre datos de prueba
portfolio_test = functions.run_trading_strategy(test_df,
                                             capital=100000,
                                             take_profit_ratio=best_params_train[0],
                                             stop_loss_ratio=best_params_train[1],
                                             position_size=best_params_train[2])

# Métricas de atribución al desempeño
MAD_train = functions.portfolio_metrics(portfolio[["timeStamp",
                                                  "Long_Entry","Exit","Position",
                                                  "Capital","Daily_Profit",
                                                  "Take_Profit","Stop_Loss","Returns",
                                                  "Cumulative_Returns"]])
MAD_test = functions.portfolio_metrics(portfolio_test[["timeStamp",
                                                  "Long_Entry","Exit","Position",
                                                  "Capital","Daily_Profit",
                                                  "Take_Profit","Stop_Loss","Returns",
                                                  "Cumulative_Returns"]])

# Visualizaciones
# Visualización de indicadores calculados sobre las series de tiempo
# Se tienen que realizar sobre el slice de la posición 1 hacia adelante porque el Aroon en 0 es 0 y hace
# que la gráfica salga espantosa
train_ind = visualizations.plot_indicators(train_df.iloc[1:,:])
test_ind = visualizations.plot_indicators(test_df.iloc[1:,:])
# Visualización de la estrategia sobre la serie de tiempo
trading_strat_train = visualizations.plot_indicators_2(portfolio.iloc[1:,:])
trading_strat_test = visualizations.plot_indicators_2(portfolio_test.iloc[1:,:])
#Visualización de la convergencia en la optimización de la estrategia
conv_graph = visualizations.plot_capital_evolution(iter_val_test)
import pandas as pd
import numpy as np
import re
from scipy.optimize import minimize

def calculate_ema(df, period = 20):
    df["EMA_" + str(period)] = df["Close"].ewm(span=period, adjust=False).mean()
    return df

def calculate_aroon_oscillator(data, periods=14):
    """
    Calcula el Aroon Oscillator en el dataframe de entrada.

    Parámetros:
        data (pd.DataFrame): El dataframe con los datos de precios.
        high_col (str): El nombre de la columna con los precios máximos para calcular Aroon Up.
        low_col (str): El nombre de la columna con los precios mínimos para calcular Aroon Down.
        periods (int): El número de períodos para calcular el Aroon Oscillator.

    Retorna:
        pd.Series: Una serie de pandas con el Aroon Oscillator calculado.
    """
    aroon_up = 100 * (data["High"].rolling(window=periods).apply(lambda x: x.argmax()) / (periods - 1))
    aroon_down = 100 * (data["Low"].rolling(window=periods).apply(lambda x: x.argmin()) / (periods - 1))
    aroon_oscillator = aroon_up - aroon_down
    data["Aroon_Oscillator_"+str(periods)] = aroon_oscillator
    return data



def stochastic_trading_strategy_v2(in_df, stop_loss, take_profit, position_size_pct, max_open_positions=2):
    df = in_df.copy()
    df['Long_Entry'] = 0
    df['Short_Entry'] = 0
    df['Exit'] = 0
    df['Position'] = 0
    open_positions = 0

    for i in range(1, len(df)):
        if open_positions < max_open_positions:
            if (df.loc[i, '%K'] > df.loc[i, '%D']) and (df.loc[i - 1, '%K'] <= df.loc[i - 1, '%D']) and \
                    (df.loc[i, '%K'] < 50) and (df.loc[i, '%D'] < 50) and (df.loc[i, 'Close'] > df.loc[i, 'EMA_20']):
                df.loc[i, 'Long_Entry'] = 1
                df.loc[i, 'Position'] = position_size_pct
                open_positions += 1

            if (df.loc[i, '%K'] < df.loc[i, '%D']) and (df.loc[i - 1, '%K'] >= df.loc[i - 1, '%D']) and \
                    (df.loc[i, '%K'] > 50) and (df.loc[i, '%D'] > 50) and (df.loc[i, 'Close'] < df.loc[i, 'EMA_20']):
                df.loc[i, 'Short_Entry'] = -1
                df.loc[i, 'Position'] = -position_size_pct
                open_positions += 1

        if open_positions > 0:
            if (df.loc[i, 'Position'] > 0 and (df.loc[i, 'Close'] >= df.loc[i, 'Close'] * (1 + take_profit) or
                                               df.loc[i, 'Close'] <= df.loc[i, 'Close'] * (1 - stop_loss))) or \
               (df.loc[i, 'Position'] < 0 and (df.loc[i, 'Close'] <= df.loc[i, 'Close'] * (1 - take_profit) or
                                               df.loc[i, 'Close'] >= df.loc[i, 'Close'] * (1 + stop_loss))):
                df.loc[i, 'Exit'] = 1
                df.loc[i, 'Position'] = 0
                open_positions -= 1

    return df

def calculate_parabolic_sar(data, high_col = "High", low_col = "Low", acceleration=0.02, maximum=0.2):
    """
    Calcula el Parabolic SAR en el dataframe de entrada.

    Parámetros:
        data (pd.DataFrame): El dataframe con los datos de precios.
        high_col (str): El nombre de la columna con los precios máximos.
        low_col (str): El nombre de la columna con los precios mínimos.
        acceleration (float): El factor de aceleración inicial.
        maximum (float): El factor de aceleración máximo.

    Retorna:
        pd.Series: Una serie de pandas con el Parabolic SAR calculado.
    """
    high = data[high_col].values
    low = data[low_col].values

    # Inicializar variables
    start = 1
    length = len(data)
    sar = np.zeros(length)
    ep = np.zeros(length)
    af = np.zeros(length)
    trend = np.ones(length)

    # Primer valor de SAR
    sar[start] = min(low[start - 1], low[start])

    # Bucle para calcular el Parabolic SAR
    for i in range(start + 1, length):
        if trend[i - 1] == 1:  # Tendencia alcista
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            if high[i] > ep[i - 1]:
                ep[i] = high[i]
                af[i] = min(af[i - 1] + acceleration, maximum)
            else:
                ep[i] = ep[i - 1]
                af[i] = af[i - 1]

            if sar[i] > low[i]:
                sar[i] = min(low[i - 1], low[i])
                trend[i] = -1
                ep[i] = low[i]
                af[i] = acceleration
        else:  # Tendencia bajista
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            if low[i] < ep[i - 1]:
                ep[i] = low[i]
                af[i] = min(af[i - 1] + acceleration, maximum)
            else:
                ep[i] = ep[i - 1]
                af[i] = af[i - 1]

            if sar[i] < high[i]:
                sar[i] = max(high[i - 1], high[i])
                trend[i] = 1
                ep[i] = high[i]
                af[i] = acceleration

    return pd.Series(sar, index=data.index)

def run_trading_strategy(data_in, capital=100000, position_size=1000, take_profit_ratio=0.01, stop_loss_ratio=0.01):
    data = data_in.copy()

    # Encontrar la columna de EMA usando una expresión regular
    ema_column = None
    for col in data.columns:
        if re.match(r'EMA_\d+', col):
            ema_column = col
            break

    if ema_column is None:
        raise ValueError("No se encontró la columna de EMA en el dataframe")

    data['Long_Entry'] = 0
    data['Exit'] = 0
    data['Position'] = 0
    data['Capital'] = capital
    data['Daily_Profit'] = 0
    data['Take_Profit'] = 0
    data['Stop_Loss'] = 0

    for i in range(1, len(data)):
        if i > 0:
            data.loc[i, 'Position'] = data.loc[i - 1, 'Position']

        if (data.loc[i, 'Close'] > data.loc[i, ema_column]) and \
           (data.loc[i, 'Parabolic_SAR'] < data.loc[i, 'Close']) and \
           (data.loc[i, 'Aroon_Oscillator_14'] > 0) and \
           (data.loc[i - 1, 'Position'] == 0):
            data.loc[i, 'Long_Entry'] = 1
            data.loc[i, 'Position'] = 1

        if (data.loc[i, 'Close'] < data.loc[i, ema_column]) and \
           (data.loc[i, 'Parabolic_SAR'] > data.loc[i, 'Close']) and \
           (data.loc[i, 'Aroon_Oscillator_14'] < 0) and \
           (data.loc[i - 1, 'Position'] == 1):
            data.loc[i, 'Exit'] = 1
            data.loc[i, 'Position'] = 0

        data.loc[i, 'Returns'] = (data.loc[i, 'Close'] / data.loc[i - 1, 'Close'] - 1)

        if data.loc[i, 'Position'] != 0:
            data.loc[i, 'Capital'] = data.loc[i - 1, 'Capital'] * (1 + data.loc[i, 'Returns'] * (position_size / capital))
        else:
            data.loc[i, 'Capital'] = data.loc[i - 1, 'Capital']

        data.loc[i, 'Daily_Profit'] = (data.loc[i, 'Capital'] - data.loc[i - 1, 'Capital']) * (position_size / capital) * 100

        if data.loc[i, 'Daily_Profit'] >= data.loc[i - 1, 'Capital'] * take_profit_ratio:
            if data.loc[i, 'Exit'] == 0 and data.loc[i - 1, 'Position'] == 1:
                data.loc[i, 'Take_Profit'] = 1
        elif data.loc[i, 'Daily_Profit'] <= -1 * data.loc[i - 1, 'Capital'] * stop_loss_ratio:
            if data.loc[i, 'Exit'] == 0 and data.loc[i - 1, 'Position'] == 1:
                data.loc[i, 'Stop_Loss'] = 1

        if (data.loc[i, 'Take_Profit'] == 1 or data.loc[i, 'Stop_Loss'] == 1) and data.loc[i - 1, 'Position'] == 1:
            data.loc[i, 'Exit'] = 1
            data.loc[i, 'Position'] = 0

        data['Cumulative_Returns'] = (data['Capital'] / capital) - 1

    return data

def best_fit_params(data, params=[0.01, 0.01, 100], verbose=False):
    objective_values = []

    bounds = [
        (0.001, 0.1),  # take_profit_ratio
        (0.001, 0.1),  # stop_loss_ratio
        (100, 10000)   # position_size
    ]

    def objective_function(params, data):
        take_profit_ratio, stop_loss_ratio, position_size = params
        result = run_trading_strategy(data, capital=100000, take_profit_ratio=take_profit_ratio, stop_loss_ratio=stop_loss_ratio, position_size=position_size)
        objective_value = -result['Capital'].iloc[-1]
        objective_values.append(objective_value)
        if verbose:
            print("Objective function value:", objective_value)
        return objective_value

    def callback(x):
        if verbose:
            print("Current parameters:", x)

    optimization_result = minimize(objective_function, params, args=(data,), bounds=bounds, method='L-BFGS-B', callback=callback)

    return optimization_result.x, objective_values


def portfolio_metrics(data, risk_free_rate=0.05):
    """
    Calcula las métricas del portafolio: Retorno Total, Retorno Promedio, Radio de Sharpe.

    Parámetros:
        data (pd.DataFrame): El dataframe con las columnas requeridas.
        risk_free_rate (float): Tasa libre de riesgo. Por defecto es 0.05.

    Retorna:
        pd.DataFrame: Un dataframe con las métricas calculadas.
    """
    initial_capital = data.loc[0, 'Capital']
    final_capital = data.loc[len(data) - 1, 'Capital']
    total_return = (final_capital / initial_capital) - 1
    average_return = data['Returns'].mean()
    
    volatility = data['Returns'].std()
    sharpe_ratio = (average_return - risk_free_rate) / volatility

    metrics = pd.DataFrame({'Total_Return': [total_return*100],
                            'Average_Return': [average_return*100],
                            'Volatility': [volatility*100],
                            'Sharpe_Ratio': [sharpe_ratio]})
    
    return metrics
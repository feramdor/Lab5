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

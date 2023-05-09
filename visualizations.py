import plotly.graph_objects as go
import re
from plotly.subplots import make_subplots
def ts_plot(df):

    # Crea una figura de Plotly con subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Close and EMA_20", "%K and %D"))

    # Agrega la serie de tiempo Close a la primera faceta
    fig.add_trace(go.Scatter(x=df['timeStamp'], y=df['Close'], mode='lines', name='Close'), row=1, col=1)

    # Agrega la serie de tiempo EMA_20 a la primera faceta
    fig.add_trace(go.Scatter(x=df['timeStamp'], y=df['EMA_20'], mode='lines', name='EMA_20'), row=1, col=1)

    # Agrega la serie de tiempo %K a la segunda faceta
    fig.add_trace(go.Scatter(x=df['timeStamp'], y=df['%K'], mode='lines', name='%K'), row=2, col=1)

    # Agrega la serie de tiempo %D a la segunda faceta
    fig.add_trace(go.Scatter(x=df['timeStamp'], y=df['%D'], mode='lines', name='%D'), row=2, col=1)

    # Actualiza los ejes y el layout de la figura
    fig.update_layout(title='Close, EMA_20, %K and %D Time Series', showlegend=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Stochastic Oscillator", row=2, col=1)

    # Muestra la figura
    return fig


def plot_indicators(data):
    """
    Crea una gráfica con facetas que muestre las series de tiempo de los indicadores y el precio.

    Parámetros:
        data (pd.DataFrame): El dataframe con los datos de precios y los indicadores calculados.
    """

    # Crear el objeto de subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Gráfico de precios, EMA y Parabolic SAR
    # EXPERIMENTAL GRAFICAR PRECIO COMO VELAS, NO USAR; QUEDA BIEN CULERO
    """
    fig.add_trace(go.Candlestick(x=data["timeStamp"],
                                 open=data["Open"],
                                  high=data['High'],
                                  low=data['Low'],
                                  close=data['Close'],
                                  name='Precio'),
                  row=1, col=1)
    """
    fig.add_trace(go.Scatter(x=data["timeStamp"],y=data["Close"], name="Precio"),row=1,col=1)

    ema_column = None
    for col in data.columns:
        if re.match(r'EMA_\d+', col):
            ema_column = col
            break

    if ema_column is not None:
        fig.add_trace(go.Scatter(x=data["timeStamp"], y=data[ema_column], name='EMA', line=dict(color='orange')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["timeStamp"], y=data['Parabolic_SAR'], mode='markers', name='Parabolic SAR', marker=dict(size=3, color='purple')), row=1, col=1)

    # Gráfico del Aroon Oscillator
    fig.add_trace(go.Bar(x=data["timeStamp"], y=data['Aroon_Oscillator_14'], name='Aroon Oscillator'), row=2, col=1)

    # Actualizar la apariencia de las gráficas
    fig.update_layout(title='Indicadores Técnicos y Precio',
                      xaxis=dict(title='timeStamp'),
                      yaxis=dict(title='Precio', domain=[0.3, 1]),
                      yaxis2=dict(title='Aroon Oscillator', domain=[0, 0.25]),
                      template='plotly_dark')

    # Mostrar la gráfica
    return fig



def plot_indicators_2(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=data["timeStamp"],y=data["Close"], name="Precio"),row=1,col=1)

    ema_column = None
    for col in data.columns:
        if re.match(r'EMA_\d+', col):
            ema_column = col
            break

    if ema_column is not None:
        fig.add_trace(go.Scatter(x=data["timeStamp"], y=data[ema_column], name='EMA', line=dict(color='orange')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["timeStamp"], y=data['Parabolic_SAR'], mode='markers', name='Parabolic SAR', marker=dict(size=3, color='purple')), row=1, col=1)

    long_entry_dates = data.loc[data["Long_Entry"] == 1, "timeStamp"]
    for date in long_entry_dates:
        fig.add_shape(type='line', x0=date, x1=date, y0=data["Close"].min(), y1=data["Close"].max(), yref='paper', xref='x', row=1, col=1, line=dict(color='green', width=1))

    exit_dates = data.loc[data["Exit"] == 1, "timeStamp"]
    for date in exit_dates:
        fig.add_shape(type='line', x0=date, x1=date, y0=data["Close"].min(), y1=data["Close"].max(), yref='paper', xref='x', row=1, col=1, line=dict(color='red', width=1))

    fig.add_trace(go.Bar(x=data["timeStamp"], y=data['Aroon_Oscillator_14'], name='Aroon Oscillator'), row=2, col=1)

    fig.update_layout(title='Indicadores Técnicos y Precio',
                      xaxis=dict(title='timeStamp'),
                      yaxis=dict(title='Precio', domain=[0.3, 1]),
                      yaxis2=dict(title='Aroon Oscillator', domain=[0, 0.25]),
                      template='plotly_dark')

    return fig
def plot_capital_evolution(capital_values):
    # Multiplicar los valores de capital por -1
    neg_capital_values = [-value for value in capital_values]

    # Crear el gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(neg_capital_values))), y=neg_capital_values, mode='lines+markers', name='Capital'))

    fig.update_layout(
        title='Evolución del Capital en Optimización',
        xaxis_title='Iteración',
        yaxis_title='Capital',
        template='plotly_dark'
    )

    return fig

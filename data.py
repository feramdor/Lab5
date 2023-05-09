import pandas as pd
import openpyxl

def data_open(file : str) -> pd.DataFrame:
    """
    Factset XLSX File Processing
    """
    df = pd.read_excel("files/"+file,"Price History",engine="openpyxl")
    df.rename(columns={"Date":"timeStamp","Mid" : "Close"},inplace=True)
    df = df[["timeStamp","Close","High","Low"]]
    return df

import pandas as pd

def data_open_2(file: str) -> pd.DataFrame:
    """
    Yahoo Finance CSV Download Processing
    """
    df = pd.read_csv("files/" + file)
    required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close"]

    if all(column in df.columns for column in required_columns):
        df.rename(columns={"Date": "timeStamp", "Adj Close": "Mid"}, inplace=True)
        return df[["timeStamp", "Open", "High", "Low", "Close", "Mid"]]
    else:
        print("Error de Lectura")
        return None
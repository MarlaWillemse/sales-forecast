import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

root = '/home/marla/Desktop/sales_forecast'

def eda(df):
    """
    Exploratory data analysis: print basic info about pandas dataframe

    :param df: pandas df
    :return: prints data summary
    """

    print("----------Top-5- Records----------")
    print(df.head(5))
    print("-----------Information-----------")
    print(df.info())
    print("-----------Data Types-----------")
    print(df.dtypes)
    print("----------Missing Values-----------")
    print(df.isnull().sum())
    print("----------Null Values-----------")
    print(df.isna().sum())
    print("----------Shape of Data----------")
    print(df.shape)


def boxplot(df, col, saveas):
    """
    Create boxplot

    :param df: pandas df
    :param col: df column
    :param saveas: image name
    :return: png image of a boxplot in reports/figures folder
    """
    plt.figure(figsize=(10, 4))
    plt.xlim(df[col].min(), df[col].max() * 1.1)
    sns.boxplot(x=df[col])
    return plt.savefig(root+f"/reports/figures/{saveas}.png")

def downcast_dtypes(df):
    """
    Reduce precision of numeric data to reduce memory cost

    :param df: pandas df
    :return: pandas df in which numeric types are either float32 or
    int16
    """

    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)

    return df

def month_and_day_from_date(df):
    """
    Create Month and Date columns from Date
    """

    df['Month'] = pd.DatetimeIndex(df['Date']).month * 1
    df['Day'] = pd.DatetimeIndex(df['Date']).day * 1
    df = df.drop('Date', axis=1)
    return df

def reconstruct_date(df):
    """
    Reconstruct date 2019-MM-DD from 'Month' and 'Day' columns
    """

    df['Year'] = '2019'
    df['Month'] = df['Month'].apply(str)
    df['Day'] = df['Day'].apply(str)
    df['Date'] = df['Year'].str \
        .cat(df['Month'], sep="-")
    df['Date'] = df['Date'].str \
        .cat(df['Day'], sep="-")
    df = df.drop('Year', axis=1)
    df = df.drop('Month', axis=1)
    df = df.drop('Day', axis=1)
    df.Date = pd.to_datetime(df.Date)
    return df


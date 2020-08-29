import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

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
    return plt.savefig(f"../reports/figures/{saveas}.png")

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
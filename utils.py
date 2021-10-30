import numpy as np
import pandas as pd

def read_dataset(path):
    df = pd.read_csv(path)
    return df

def create_features_target(df,id_col,pred_col):
    X = df.drop(id_col)
    y = df[pred_col]
    return X,y


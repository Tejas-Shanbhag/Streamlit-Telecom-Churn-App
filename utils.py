import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics
import matplotlib.pyplot as plt


def read_dataset(path):
    df = pd.read_csv(path)
    return df

def create_features_target(df,id_col,pred_col):
    X = df.drop([id_col,pred_col],axis=1)
    y = df[pred_col]
    return X,y

# Defining the function for the ROC curve
def draw_roc( test_actual , test_probs,algo_name):
    plt.figure(figsize=(10,10))
    fpr, tpr, thresholds = metrics.roc_curve( test_actual, test_probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( test_actual, test_probs )
    plt.plot( fpr, tpr, label='ROC curve (area = %0.4f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('AUC on the Test set({})'.format(algo_name))
    plt.legend(loc="lower right")  
    plt.grid()
    plt.savefig("plots/roc.png")

    return None


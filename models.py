import streamlit as st
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def hyperparameters(model_name):
    params = {}
    if model_name == 'Logistic Regression':
        params['C'] = st.sidebar.slider("C",0.5,10.0,step=0.5)
    if model_name == 'LightGBM':
        params['n_estimators'] = st.sidebar.slider("Number of trees",50,150,step=10)
        params['max_depth'] = st.sidebar.slider("Depth of the Tree",1,15,step=1)
    return params


def model(model_name,params):
    if model_name == 'Logistic Regression':
        estimator = LogisticRegression(C = params['C'],random_state=42)
    if model_name == 'LightGBM':
        estimator = lgb.LGBMClassifier(n_estimators = params['n_estimators'],max_depth = params['max_depth'] ,random_state=42)
    return estimator
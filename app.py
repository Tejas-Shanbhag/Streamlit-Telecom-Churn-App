import streamlit as st
from utils import read_dataset
import xgboost as xgb
import os


st.title("Telecom Churn App")
st.write("  ")
st.image('images/img.png')

#Create a select slider for the dataset
dataset_name = st.sidebar.selectbox('Choose a Dataset',os.listdir('datasets/'))

# Read the dataset into a pandas dataframe
df = read_dataset('datasets/' + dataset_name)

id_col   = st.sidebar.selectbox('Choose an ID column',df.columns,index=0)
pred_col = st.sidebar.selectbox('Choose a Prediction column',df.columns,index=len(df.columns)-1)
model    = st.sidebar.selectbox("Choose a model",['Logistic Regression','Random Forest','XGBoost'])

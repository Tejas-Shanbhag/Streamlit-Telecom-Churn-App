import streamlit as st
import numpy as np
import pandas as pd
from utils import draw_roc, read_dataset,create_features_target,draw_roc
from models import hyperparameters,model
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,plot_confusion_matrix,roc_auc_score
import os


st.title("Telecom Churn App")
st.write("  ")
st.image('images/img.png')

#Create a select slider for the dataset
dataset_name = st.sidebar.selectbox('Choose a Dataset',os.listdir('datasets/'))

# Read the dataset into a pandas dataframe
df = read_dataset('datasets/' + dataset_name)
st.write("")
st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns")

id_col   = st.sidebar.selectbox('Choose an ID column',df.columns,index=0)
pred_col = st.sidebar.selectbox('Choose a Prediction column',df.columns,index=len(df.columns)-1)
model_name = st.sidebar.selectbox("Choose a model",['LightGBM','Logistic Regression'])


st.sidebar.subheader("Choose Model Hyperparameterss")
params = hyperparameters(model_name)
clf = model(model_name,params)

X,y = create_features_target(df,id_col,pred_col)
X= pd.get_dummies(X)

#Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
button_var = st.sidebar.button("Make Predictions")

y_test_list = y_test.values.tolist()
y_test_bool = [1 if x=='Yes' else 0 for x in y_test_list]


if button_var:
    with st.spinner("Taining the Model....."):
        #train the model
        clf.fit(X_train,y_train)

        #make the predictions on the test set
        preds = clf.predict(X_test)
        pred_probs = clf.predict_proba(X_test)[:,1]
        
        st.header("Model Performance")
        st.write("")
        st.write(f"The Accuracy of the  Model is {round(accuracy_score(y_test,preds)*100,2)}%.")
        st.write(f"The AUC Score of the Model is {round(roc_auc_score(y_test,pred_probs),2)}")


        fig, ax = plt.subplots(figsize=(10, 10))
        plot_confusion_matrix(clf, X_test, y_test, ax=ax)    
        plt.savefig('plots/cm.png')
        draw_roc( y_test_bool , pred_probs,clf)

        col1,col2 = st.beta_columns(2)
        col1.header("Confusion Matrix")
        col1.image("plots/cm.png")
        col2.header("ROC AUC Curve")
        col2.image("plots/roc.png")



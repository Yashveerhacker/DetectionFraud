

import streamlit as st

import pandas as pd

import numpy as np

import requests

import matplotlib.pyplot as plt

import seaborn as sns



st.title(
"Fraud Detection Dashboard"
)



st.write(
"XGBoost + FastAPI Machine Learning System"
)



# ---------------------------
# Prediction Section
# ---------------------------


st.header(
"Transaction Prediction"
)



features=[]


for i in range(30):

    value = st.number_input(

        "Feature "+str(i),

        value=0.0

    )

    features.append(value)




if st.button("Predict Fraud"):


    response=requests.post(

        "http://127.0.0.1:8000/predict",

        json={

            "features":features

        }

    )


    result=response.json()



    st.success(

        result["prediction"]

    )


    st.metric(

        "Fraud Probability",

        result["fraud_probability"]

    )



# ---------------------------
# Dataset Graphs
# ---------------------------


st.header(
"Dataset Analysis"
)



df=pd.read_csv(
"creditcard.csv"
)



# Fraud count graph


st.subheader(
"Fraud vs Normal Transactions"
)


fig,ax=plt.subplots()


sns.countplot(

    x="Class",

    data=df,

    ax=ax

)


st.pyplot(fig)




# Amount distribution


st.subheader(
"Transaction Amount Distribution"
)



fig2,ax2=plt.subplots()



sns.histplot(

    df["Amount"],

    bins=50,

    ax=ax2

)



st.pyplot(fig2)





# Correlation graph


st.subheader(
"Correlation Heatmap"
)



fig3,ax3=plt.subplots(

    figsize=(10,6)

)



sns.heatmap(

    df.corr(),

    ax=ax3

)



st.pyplot(fig3)


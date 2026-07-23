import os
import threading
import time
import subprocess


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import joblib


from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


from fastapi import FastAPI
from pydantic import BaseModel



# ==========================================
# GLOBAL VARIABLES
# ==========================================

model = None
scaler = None

dataset = None



# ==========================================
# TRAIN MODEL
# ==========================================

def train_model():

    global model, scaler, dataset


    print("Loading Dataset...")


    dataset = pd.read_csv(
        "creditcard.csv"
    )


    X = dataset.drop(
        "Class",
        axis=1
    )


    y = dataset["Class"]



    X_train,X_test,y_train,y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    scaler = StandardScaler()



    X_train = scaler.fit_transform(
        X_train
    )


    X_test = scaler.transform(
        X_test
    )



    model = XGBClassifier(

        n_estimators=200,

        learning_rate=0.05,

        max_depth=5,

        random_state=42

    )



    model.fit(

        X_train,

        y_train

    )



    prediction = model.predict(
        X_test
    )



    print(

        classification_report(

            y_test,

            prediction

        )

    )



    joblib.dump(

        model,

        "fraud_model.pkl"

    )


    joblib.dump(

        scaler,

        "scaler.pkl"

    )


    print(
        "Model Training Completed"
    )





# ==========================================
# FASTAPI
# ==========================================


api = FastAPI(

    title="Fraud Detection API"

)



class Transaction(BaseModel):

    features:list





@api.get("/")
def home():

    return {

        "message":

        "Fraud Detection API Running"

    }




@api.post("/predict")
def predict(transaction:Transaction):


    values = np.array(

        transaction.features

    ).reshape(1,-1)



    values = scaler.transform(
        values
    )



    prediction = model.predict(
        values
    )



    probability = model.predict_proba(
        values
    )



    if prediction[0] == 1:

        result="Fraud"

    else:

        result="Normal"



    return {

        "prediction":result,

        "fraud_probability":

        float(probability[0][1])

    }





def run_fastapi():

    import uvicorn


    uvicorn.run(

        api,

        host="127.0.0.1",

        port=8000

    )





# ==========================================
# STREAMLIT FILE
# ==========================================


def create_dashboard():



    code = r'''

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

'''




    with open(

        "dashboard.py",

        "w",

        encoding="utf-8"

    ) as f:

        f.write(code)





def run_streamlit():


    subprocess.run(

        [

            "streamlit",

            "run",

            "dashboard.py"

        ]

    )





# ==========================================
# MAIN
# ==========================================


if __name__=="__main__":


    train_model()



    create_dashboard()



    print(
        "Starting FastAPI..."
    )


    thread = threading.Thread(

        target=run_fastapi

    )


    thread.start()



    time.sleep(5)



    print(
        "Starting Streamlit..."
    )


    run_streamlit()
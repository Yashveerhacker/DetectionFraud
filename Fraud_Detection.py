# ============================================
# CREDIT CARD FRAUD DETECTION PROJECT
# ============================================


# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Machine Learning Libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier


# Evaluation Metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Handling Imbalanced Data

from imblearn.over_sampling import SMOTE


# Save Model

import joblib



# ============================================
# 1. LOAD DATASET
# ============================================


df = pd.read_csv(
    "creditcard.csv"
)


print("First 5 rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)



# ============================================
# 2. DATA INFORMATION
# ============================================


print("\nDataset Information:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())



# ============================================
# 3. CHECK FRAUD DISTRIBUTION
# ============================================


print("\nClass Distribution:")
print(df['Class'].value_counts())



# Visualization

plt.figure(figsize=(6,4))

sns.countplot(
    x='Class',
    data=df
)

plt.title(
    "Normal vs Fraud Transactions"
)

plt.show()



# ============================================
# 4. DATA PREPROCESSING
# ============================================


# Separate Features and Target

X = df.drop(
    'Class',
    axis=1
)


y = df['Class']



# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Feature Scaling

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)



# ============================================
# 5. HANDLE IMBALANCED DATA USING SMOTE
# ============================================


print("\nBefore SMOTE:")

print(
    y_train.value_counts()
)



smote = SMOTE(
    random_state=42
)


X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)



print("\nAfter SMOTE:")

print(
    pd.Series(y_train).value_counts()
)



# ============================================
# 6. LOGISTIC REGRESSION MODEL
# ============================================


log_model = LogisticRegression(
    max_iter=1000
)


log_model.fit(
    X_train,
    y_train
)



log_prediction = log_model.predict(
    X_test
)



print("\n===== Logistic Regression =====")


print(
"Accuracy:",
accuracy_score(
    y_test,
    log_prediction
)
)


print(
"Precision:",
precision_score(
    y_test,
    log_prediction
)
)


print(
"Recall:",
recall_score(
    y_test,
    log_prediction
)
)


print(
"F1 Score:",
f1_score(
    y_test,
    log_prediction
)
)



print(
classification_report(
    y_test,
    log_prediction
)
)




# ============================================
# 7. RANDOM FOREST MODEL
# ============================================



rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)



rf_model.fit(
    X_train,
    y_train
)



rf_prediction = rf_model.predict(
    X_test
)



print("\n===== Random Forest =====")



print(
"Accuracy:",
accuracy_score(
    y_test,
    rf_prediction
)
)



print(
"Precision:",
precision_score(
    y_test,
    rf_prediction
)
)



print(
"Recall:",
recall_score(
    y_test,
    rf_prediction
)
)



print(
"F1 Score:",
f1_score(
    y_test,
    rf_prediction
)
)



print(
classification_report(
    y_test,
    rf_prediction
)
)




# ============================================
# 8. CONFUSION MATRIX
# ============================================


cm = confusion_matrix(
    y_test,
    rf_prediction
)



plt.figure(figsize=(6,4))


sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)



plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "Confusion Matrix - Random Forest"
)


plt.show()



# ============================================
# 9. SAVE MODEL
# ============================================



joblib.dump(
    rf_model,
    "fraud_detection_model.pkl"
)



joblib.dump(
    scaler,
    "scaler.pkl"
)



print(
"\nModel Saved Successfully!"
)
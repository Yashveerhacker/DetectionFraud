import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_and_save():
    print("Loading dataset...")
    df = pd.read_csv("creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Class imbalance compensation
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost Model...")
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))

    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and Scaler successfully saved.")

if __name__ == "__main__":
    train_and_save()
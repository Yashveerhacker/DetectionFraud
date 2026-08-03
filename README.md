# 💳 Credit Card Fraud Detection System

A machine learning project that detects fraudulent credit card transactions using the **XGBoost** algorithm. The project automatically trains the model, launches a FastAPI prediction service, and opens a Streamlit dashboard for real-time fraud prediction and data visualization.

---

# 🚀 Features

* Train an XGBoost fraud detection model
* Automatic feature scaling using StandardScaler
* Save trained model using Joblib
* Real-time fraud prediction through FastAPI
* Interactive Streamlit dashboard
* Dataset analysis with graphs
* REST API for integration with other applications

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* FastAPI
* Uvicorn
* Streamlit
* Matplotlib
* Seaborn

---

# 📂 Project Structure

```text
Fraud-Detection/
│
├── creditcard.csv
├── fraud_model.pkl
├── scaler.pkl
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

### Features

* Time
* V1 – V28 (PCA transformed features)
* Amount
* Class

Target values:

* **0** → Normal Transaction
* **1** → Fraudulent Transaction

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Fraud-Detection.git
```

Move to the project folder

```bash
cd Fraud-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Simply execute:

```bash
python main.py
```

The program will automatically:

1. Load the dataset.
2. Train the XGBoost model.
3. Save the model (`fraud_model.pkl`).
4. Save the scaler (`scaler.pkl`).
5. Start the FastAPI server.
6. Generate the Streamlit dashboard.
7. Launch the dashboard in your browser.

---

# 🌐 API Endpoints

### Home

```
GET /
```

Response

```json
{
  "message": "Fraud Detection API Running"
}
```

### Predict Transaction

```
POST /predict
```

Request

```json
{
  "features":[
    0.0,
    0.0,
    0.0
  ]
}
```

Response

```json
{
  "prediction":"Normal",
  "fraud_probability":0.0012
}
```

---

# 📈 Dashboard Features

The Streamlit dashboard includes:

* Real-time fraud prediction
* Fraud probability display
* Fraud vs Normal transaction count
* Transaction amount distribution
* Correlation heatmap of all features

---

# 🤖 Machine Learning Model

Algorithm:

* XGBoost Classifier

Preprocessing:

* StandardScaler

Train/Test Split:

* 80% Training
* 20% Testing

Evaluation:

* Classification Report
* Confusion Matrix

---

# 📦 Output Files

After training, the following files are created:

* `fraud_model.pkl`
* `scaler.pkl`
* `dashboard.py`

---

# 🔮 Future Improvements

* Docker deployment
* Cloud deployment (AWS/Azure/GCP)
* Real-time payment gateway integration
* Model monitoring
* SHAP Explainable AI
* Improved dashboard with Power BI

---

# 👨‍💻 Author

**Yash Veer**

B.Tech (Electronics and Communication Engineering)

The LNM Institute of Information Technology (LNMIIT), Jaipur

---

# 📜 License

This project is intended for educational and learning purposes.

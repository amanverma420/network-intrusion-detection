# 🛡️ Enterprise Network Intrusion Detection System (NIDS)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://uidashboardpy.streamlit.app/)
[![FastAPI API](https://img.shields.io/badge/FastAPI-Backend%20Engine-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://uidashboardpy.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

An enterprise-grade, high-performance Network Intrusion Detection System (NIDS) powered by a dual-heuristic classifier. This project leverages both a classic **Random Forest Classifier** and a **Deep Learning 1D-CNN (Convolutional Neural Network)** to analyze and isolate malicious network connection packets in real-time.

🔗 **Access the Live Web Dashboard:** [https://uidashboardpy.streamlit.app/](https://uidashboardpy.streamlit.app/)

---

## 📂 Project Structure

```text
network-intrusion-detection/
├── data/                       # Raw datasets and lookup configuration resources (gitignored)
│   ├── KDDCup Data 10 Percent (Edited).csv
│   ├── KDDCup Data 10 Percent.csv
│   ├── Latest.csv
│   ├── Workshop Dataset.csv
│   ├── kddcup.txt
│   └── training_attack_types.txt
├── models/                     # Serialized model pipelines and binary weights (version controlled)
│   ├── cnn_model.keras         # Trained 1D-CNN Keras neural network weights
│   ├── random_forest_model.pkl # Trained Random Forest model binary
│   ├── encoders.pkl            # LabelEncoder mappings for categorical features
│   ├── feature_columns.pkl     # Sequence configuration list for input schema alignment
│   └── scaler.pkl              # StandardScaler normalization matrices
├── app.py                      # RESTful API interface powered by FastAPI
├── ui_dashboard.py             # Sleek dashboard user interface built with Streamlit
├── train_and_save.py           # End-to-end model training, scaling, and serialization script
├── Dockerfile                  # Production container assembly specification
├── requirements.txt            # Managed Python package dependency list
└── README.md                   # Enterprise system documentation
```

---

## ⚡ Key Features

* **Dual Heuristics Engine**: Parallel evaluation using a highly optimized Random Forest Classifier (99.76% accuracy) alongside a deep 1D-CNN.
* **Granular Telemetry Inputs**: Intuitive categorizations spanning **Basic Connection Headers**, **Content Flags**, and **Statistical Traffic Profiles** to run predictive diagnostics.
* **Class Confidence Metrics**: Visualizes prediction probability distributions across network profiles (e.g. normal, portsweep, neptune, smurf, etc.).
* **Production-Ready APIs**: Serves instant JSON predictions via FastAPI endpoints, containerized cleanly inside Docker.

---

## 🚀 Getting Started

### 1. Local Setup
Clone the repository and set up a virtual environment:
```bash
# Clone the repository
git clone https://github.com/amanverma420/network-intrusion-detection.git
cd network-intrusion-detection

# Setup Python Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Dashboard
Launch the interactive visual dashboard on your local machine:
```bash
streamlit run ui_dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the FastAPI Service
Launch the microservice API for remote integrations:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
* **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Prediction Endpoint:** `POST http://localhost:8000/predict`

---

## 🧠 Model Training Pipeline
To retrain the classification models using your own CSV logs:
1. Place your dataset inside the `data/` directory (e.g., as `data/Latest.csv`).
2. Execute the pipeline script:
```bash
python train_and_save.py
```
This runs validation splits, standardizes numerical features, encodes categories, trains both models, and saves the binary weights into the `models/` directory.

---

## 🐳 Containerization & Deployment

### Run via Docker
To build and run the API service inside an isolated Docker container:
```bash
# Build Docker image
docker build -t nids-api .

# Run container exposing port 8000
docker run -p 8000:8000 nids-api
```

### Cloud Deployment
This project is continuously deployed to **Streamlit Community Cloud** synced directly from the `main` branch. 
* To spin up a mirror, log in to [Streamlit Share](https://share.streamlit.io/), link this repository, and specify the **Python Version as 3.11 or 3.12** inside the Advanced Settings panel.

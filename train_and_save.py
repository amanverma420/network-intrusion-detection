import os
import sys

# Optimize speed: Skip hardware scanning and suppress log spew
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

print("⏳ Loading and clearing dataset discrepancies...")

# 1. Load Data
dataset = pd.read_csv("Latest.csv")
dataset.drop_duplicates(inplace=True)

# 2. Drop rare attack profiles BEFORE encoding so numbering remains sequential
connection_counts = dataset["connection"].value_counts()
rare_classes = connection_counts[connection_counts < 2].index

if len(rare_classes) > 0:
    print(f"⚠️ Dropping rare attack profiles with less than 2 records: {list(rare_classes)}")
    dataset = dataset[~dataset["connection"].isin(rare_classes)]

# 3. Encode Categorical Columns cleanly from 0 upwards
obj_cols = dataset.select_dtypes(include=["object", "string"]).columns
encoders = {}
for col in obj_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))
    encoders[col] = le

# Save encoder mappings
joblib.dump(encoders, "encoders.pkl")

# 4. Split Target and Features
X = dataset.drop(columns=["connection"])
y = dataset["connection"]
num_classes = len(np.unique(y))

# Save structural configurations 
joblib.dump(list(X.columns), "feature_columns.pkl")

# Safe stratification split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Fit & Save Scaler
print("⚖️ Normalizing and scaling structural data features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# 6. Train & Save Random Forest Model
print("🌲 Fitting Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, "random_forest_model.pkl")

# 7. Train & Save Convolutional Neural Network (CNN)
print("🧠 Building and fitting Deep Learning CNN Model...")
X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)

# Fixed Input(shape) setup to eliminate Keras 3 warnings
cnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(filters=32, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn_model.fit(X_train_cnn, y_train, epochs=3, batch_size=64, verbose=1)
cnn_model.save("cnn_model.keras")

print("🎉 Setup finished successfully! All serialization assets are compiled.")
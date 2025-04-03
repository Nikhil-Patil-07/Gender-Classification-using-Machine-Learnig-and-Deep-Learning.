# Open_CV_Project

Link for Dataset: - https://susanqq.github.io/UTKFace/

# Gender Classification Using Machine Learning & Deep Learning

## Features
- Pre-trained **ML models (`.pkl`)** and **DL models (`.keras`)** for gender classification.
- Uses **OpenCV** for image processing.
- Extracts deep features using **MobileNetV2**.
- Predicts using multiple models with **majority voting**.

## Project Structure
/gender_classification_project
│── dataset/                		# Folder containing images
│── model/                   		# Trained models (SVM, RF, XGBoost, CNN)
│   ├── svm_model.pkl        		# Trained SVM model
│   ├── rf_model.pkl         		# Trained Random Forest model
│   ├── xgb_model.pkl        		# Trained XGBoost model
│   ├── feature_extractor.keras 	# Trained MobileNetV2 feature extractor
│   ├── cnn_model.keras        	# Trained CNN model
│── predict_function.pkl            # Main script for running predictions
│── requirements.txt         		# Dependencies file
│── README.md                		# Documentation file

## Implementation
-- Upload all the .pkl and .keras file into the Home of Jupyter Notebook
-- Call all the .pkl and .keras file into new notebook
-- Apply prediction function by giving any image path from your device

## Code
''
from tensorflow.keras.models import load_model  # Direct import
cnn_model = load_model("gender_cnn_model.keras", compile=False)
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

import pickle       # It is built in python library therefore we cannot specify in requirements.txt

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

feature_extractor = load_model("feature_extractor.keras")

import pickle
with open("predict_function.pkl", "rb") as f:
    loaded_function = pickle.load(f)
''

-- by calling function and giving path of image as input you get your predicted output


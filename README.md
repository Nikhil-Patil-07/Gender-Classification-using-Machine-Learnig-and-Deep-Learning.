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

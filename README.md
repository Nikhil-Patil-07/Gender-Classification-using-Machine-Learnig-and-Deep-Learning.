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
│── Main.py                         # Main script for running predictions
│── requirements.txt         		# Dependencies file
│── README.md                       # Documentation file
|── Trail images                    # Images to test the model prediction             		

## Implementation
-- First run the requirements.txt by providing the filr path {'pip install -r file_path'}
-- Then run Main.py file by providing path of file {'python file_path'}
-- Provide image path without quotes and then you will get prediction for uploaded image


### Repository Link
🔗 [GitHub Repository](https://github.com/Nikhil-Patil-07/Open_CV_Project)
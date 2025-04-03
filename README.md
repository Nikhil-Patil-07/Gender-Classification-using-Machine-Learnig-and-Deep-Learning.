# Gender Classification Using Machine Learning & Deep Learning

Link for Dataset: -https://susanqq.github.io/UTKFace/

## About Dataset
  The UTKFace dataset, available on Susanqq's GitHub page, is a large-scale face dataset containing over 20,000 images labeled with age, gender, and ethnicity information. Each image is named in the format age_gender_ethnicity.jpg, where the second value represents gender (0 for male and 1 for female). This dataset is widely used for tasks such as age estimation, gender classification, and ethnicity recognition. Given its diversity in age range (0–116 years) and ethnicity labels (White, Black, Asian, Indian, and Others), it serves as a valuable resource for training and evaluating machine learning models in facial analysis. For your gender classification project, you can extract the gender labels and preprocess the dataset accordingly to achieve high-accuracy predictions.

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
-- Store all the .pkl and .keras and only Main.py files in one folder and other remaining files outside the folder
-- Then run the requirements.txt by providing the filr path {'pip install -r file_path'} in terminal 
-- Then run Main.py file by providing path of file {'python file_path'} in terminal 
-- Provide image path without quotes and then you will get prediction for uploaded image


### Repository Link
🔗 [GitHub Repository](https://github.com/Nikhil-Patil-07/Open_CV_Project)

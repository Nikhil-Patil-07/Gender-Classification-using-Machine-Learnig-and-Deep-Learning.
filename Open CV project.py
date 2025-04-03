#!/usr/bin/env python
# coding: utf-8

# ### **Reading all the images**

# In[77]:


import os
import pandas as pd
import pickle
import numpy as np
import cv2
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from xgboost import XGBClassifier
from skimage.feature import local_binary_pattern

# Load and preprocess dataset
data_dir = "D:/open cv dataset/UTKFace"  # Update with dataset path
img_size = 128  # Updated to a valid size

data = []
labels = []


# In[3]:


for img_name in os.listdir(data_dir):
    path = os.path.join(data_dir, img_name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (128, 128, 1)
    data.append(img)
    label = 1 if int(img_name.split("_")[1]) == 0 else 0  # 1=Male, 0=Female
    labels.append(label)

data = np.array(data, dtype=np.float32)  # Ensure correct format
labels = np.array(labels, dtype=np.int32)


# ### **Checking For Balance or Imbalance Data**

# In[198]:


import os

# Path to your dataset folder
dataset_path = r"D:\open cv dataset\UTKFace"

# Initialize counters
male_count = 0
female_count = 0

# Loop through all images in the folder
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
        try:
            gender = int(filename.split("_")[1])  # Extract gender from filename
            if gender == 0:
                male_count += 1
            elif gender == 1:
                female_count += 1
        except (IndexError, ValueError):
            print(f"Skipping file: {filename}")  # Handle any incorrectly named files

# Print results
print(f"Total Male Images: {male_count}")
print(f"Total Female Images: {female_count}")

# Check class imbalance
total_images = male_count + female_count
print(f"Male Percentage: {male_count / total_images * 100:.2f}%")
print(f"Female Percentage: {female_count / total_images * 100:.2f}%")


# ### **Splitting data or images**

# In[4]:


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[75]:


# Save as .pkl files (Useful for mixed data types)
with open("train_data.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)

with open("test_data.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

print("Datasets saved successfully!")


# In[5]:


# Define a CNN model for grayscale images
input_layer = Input(shape=(img_size, img_size, 1))  # Updated input shape
base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = True  # Unfreeze top layers for fine-tuning


# ### **Training CNN model**

# In[6]:


# Convert grayscale to 3 channels inside the model
x = Conv2D(3, (1, 1), padding='same', activation='relu')(input_layer)  # 1x1 Conv to match RGB input
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)


# In[7]:


cnn_model = Model(inputs=input_layer, outputs=out)
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# In[8]:


# Train CNN model
cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# In[9]:


import matplotlib.pyplot as plt

# Sample values (replace with your actual training history values)
epochs = range(1, 11)
train_accuracy = [0.7643, 0.9038, 0.9326, 0.9514, 0.9627, 0.9735, 0.9765, 0.9834, 0.9804, 0.9872]
val_accuracy = [0.5240, 0.5283, 0.5650, 0.7155, 0.7309, 0.7564, 0.7769, 0.7986, 0.8747, 0.8705]
train_loss = [0.4876, 0.2347, 0.1702, 0.1228, 0.1005, 0.0753, 0.0651, 0.0477, 0.0535, 0.0402 ]
val_loss = [1.3169, 1.3075, 1.0744, 0.5672, 0.6716, 0.6405, 0.5643, 1.3274, 0.5606, 0.5219]

# Plot Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label="Train Accuracy", marker='o')
plt.plot(epochs, val_accuracy, label="Validation Accuracy", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Sample values (replace with actual values)
epochs = np.arange(1, 11)
train_accuracy = [0.7643, 0.9038, 0.9326, 0.9514, 0.9627, 0.9735, 0.9765, 0.9834, 0.9804, 0.9872]
val_accuracy = [0.5240, 0.5283, 0.5650, 0.7155, 0.7309, 0.7564, 0.7769, 0.7986, 0.8747, 0.8705]
train_loss = [0.4876, 0.2347, 0.1702, 0.1228, 0.1005, 0.0753, 0.0651, 0.0477, 0.0535, 0.0402 ]
val_loss = [1.3169, 1.3075, 1.0744, 0.5672, 0.6716, 0.6405, 0.5643, 1.3274, 0.5606, 0.5219]

bar_width = 0.2  # Width of the bars
index = np.arange(len(epochs))  # X-axis positions

plt.figure(figsize=(12, 6))

# Bar chart for accuracy
plt.subplot(1, 2, 1)
plt.bar(index - bar_width, train_accuracy, bar_width, label="Train Accuracy", color='blue')
plt.bar(index, val_accuracy, bar_width, label="Validation Accuracy", color='green')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.xticks(index, epochs)  # Show epoch numbers
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Bar chart for loss
plt.subplot(1, 2, 2)
plt.bar(index - bar_width, train_loss, bar_width, label="Train Loss", color='red')
plt.bar(index, val_loss, bar_width, label="Validation Loss", color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.xticks(index, epochs)  # Show epoch numbers
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[11]:


# Extract Features from CNN for ML models
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-4].output)  # Extract from GlobalAveragePooling2D layer
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)


# ### **Fitting ML model**

# In[12]:


# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_features, y_train)



# In[13]:


#Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_features, y_train)


# In[14]:


# Train XGBoost model (Fixed)
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)  # Removed use_label_encoder
xgb_model.fit(X_train_features, y_train)


# ### **Evaluating model**

# In[15]:


# Function to evaluate and visualize results
def evaluate_model(model, X_test_features, y_test, model_name):
    y_pred = model.predict(X_test_features)
    y_prob = model.predict_proba(X_test_features)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob) if np.any(y_prob) else (None, None, None)
    auc_score = auc(fpr, tpr) if fpr is not None else 0.0
    
    print(f"{model_name} Performance:")
    print(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    plt.figure(figsize=(5,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Female", "Male"], yticklabels=["Female", "Male"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    
    if fpr is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.show()



# In[16]:


# Evaluate models
evaluate_model(svm_model, X_test_features, y_test, "SVM")


# In[17]:


evaluate_model(rf_model, X_test_features, y_test, "Random Forest")


# In[18]:


evaluate_model(xgb_model, X_test_features, y_test, "XGBoost")


# In[19]:


import matplotlib.pyplot as plt

# Data
model_names = ['Random Forest', 'XGBoost Classifier', 'SVC']
accuracies = [0.90, 0.91, 0.90]

# Colors
colors = ['#FF6B6B', '#4ECDC4', '#556270']

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black')

# Add accuracy values on top
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f"{bar.get_height()*100:.2f}%", ha='center', fontsize=12, fontweight='bold')

# Style
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xticks(rotation=15, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()


# ### **Saving and loading Model**

# In[86]:


cnn_model.save("gender_cnn_model.keras")  # Saves full model


# In[81]:


import pickle

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)


# In[83]:


feature_extractor.save("feature_extractor.keras")  # New recommended format


# In[199]:


from tensorflow.keras.models import load_model  # Direct import
cnn_model = load_model("gender_cnn_model.keras", compile=False)
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[200]:


with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)


# In[201]:


feature_extractor = load_model("feature_extractor.keras")


# In[202]:


print(feature_extractor.summary())  # Check if feature extractor is loaded properly


# ### **Create User define function with Ensemble modeling**

# In[203]:


def predict_single_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize for visualization
    display_size = (img.shape[1] // 2, img.shape[0] // 2)
    img_resized = cv2.resize(img, display_size, interpolation=cv2.INTER_AREA)

    # Preprocess image for model input
    img_model = cv2.resize(img, (128, 128))
    img_model = cv2.cvtColor(img_model, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img_model = img_model / 255.0  # Normalize
    img_model = np.expand_dims(img_model, axis=-1)  # Shape (128, 128, 1)
    img_model = np.expand_dims(img_model, axis=0)  # Add batch dimension

    # Extract features (grayscale input)
    feature_vector = feature_extractor.predict(img_model)
    feature_vector = feature_vector.reshape(1, -1)

    # Predict with ML models
    gender_svm = svm_model.predict(feature_vector)[0]
    gender_rf = rf_model.predict(feature_vector)[0]
    gender_xgb = xgb_model.predict(feature_vector)[0]

    # Majority voting
    votes = [gender_svm, gender_rf, gender_xgb]
    gender_final = "Male" if sum(votes) > 1 else "Female"

    # Assign color
    title_color = "blue" if gender_final == "Male" else "green"

    # Display image
    plt.figure(figsize=(5, 5))
    plt.imshow(img_resized)
    plt.axis("off")
    plt.title(f"Predicted Gender: {gender_final}", fontsize=12, color=title_color)
    plt.show()

    return f"Predicted Gender: {gender_final}"


# ### **Predicting for male images while actual gender is male**

# In[204]:


predict_single_image('Downloads/images/male/men1.jpg')


# In[205]:


predict_single_image("Downloads/images/male/male9.jfif")


# In[206]:


predict_single_image("Downloads/images/male/male8.jpg")


# In[207]:


predict_single_image("Downloads/images/male/male7.jpg")


# In[208]:


predict_single_image("Downloads/images/male/male6.jpg")


# In[209]:


predict_single_image("Downloads/images/male/male5.jpg")


# In[210]:


predict_single_image("Downloads/images/male/male4.jpeg")


# In[211]:


predict_single_image("Downloads/images/male/male3.jfif")


# In[212]:


predict_single_image("Downloads/images/male/male2.jpg")


# In[213]:


predict_single_image("Downloads/images/male/male15.jpg")


# In[214]:


predict_single_image("Downloads/images/male/male14.jfif")


# In[215]:


predict_single_image("Downloads/images/male/male13.jpg")


# In[216]:


predict_single_image("Downloads/images/male/male12.jfif")


# In[217]:


predict_single_image("Downloads/images/male/male11.jpg")


# In[218]:


predict_single_image("Downloads/images/male/male10.jpg")


# ### **Predicting for female images while actual gender is female**

# In[219]:


predict_single_image("Downloads/images/female/feamel2.jpg")


# In[220]:


predict_single_image("Downloads/images/female/female1.jpg")


# In[221]:


predict_single_image("Downloads/images/female/female10.jpg")


# In[222]:


predict_single_image("Downloads/images/female/female11.jpg")


# In[223]:


predict_single_image("Downloads/images/female/female12.jfif")


# In[224]:


predict_single_image("Downloads/images/female/female13.jpg")


# In[225]:


predict_single_image("Downloads/images/female/female14.jfif")


# In[226]:


predict_single_image("Downloads/images/female/female15.jpg")


# In[227]:


predict_single_image("Downloads/images/female/female3.jpg")


# In[228]:


predict_single_image("Downloads/images/female/female4.jpg")


# In[229]:


predict_single_image("Downloads/images/female/female5.jpg")


# In[230]:


predict_single_image("Downloads/images/female/female6.jfif")


# In[231]:


predict_single_image("Downloads/images/female/female7.jpg")


# In[232]:


predict_single_image("Downloads/images/female/female8.jfif")


# In[233]:


predict_single_image("Downloads/images/female/female9.jpg")


# ### **Creating user define function for predicting count for men and female in the image with Ensemble modeling** 

# In[234]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import pickle


# In[235]:


def predict_group_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MTCNN

    # Detect faces
    detections = detector.detect_faces(img)
    if not detections:
        print("No faces detected in the image.")
        return 0, 0  

    male_count, female_count = 0, 0

    # Refined weights (to prevent bias)
    weights = {"svm": 1.0, "rf": 1.0, "xgb": 0.98}

    for detection in detections:
        if 'box' in detection:
            x, y, w, h = detection['box']
        else:
            print("No 'box' key found in detection output:", detection)
            continue

        # Ensure valid bounding box coordinates
        x, y = max(0, x), max(0, y)
        w, h = max(0, w), max(0, h)

        # Extract face ROI and preprocess
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = face / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)

        # Extract features using feature extractor
        face_features = feature_extractor.predict(face)
        face_features = face_features.reshape(1, -1)

        # Get predictions from models
        svm_pred = svm_model.predict(face_features)[0]
        rf_pred = rf_model.predict(face_features)[0]
        xgb_pred = xgb_model.predict(face_features)[0]

        # Compute weighted scores
        male_score = (svm_pred * weights["svm"]) + (rf_pred * weights["rf"]) + (xgb_pred * weights["xgb"])
        female_score = (3 - male_score)  # Binary classification adjustment

        # Final gender classification using weighted scores
        gender_final = 1 if male_score > female_score else 0

        if gender_final == 1:
            male_count += 1
            label = "Male"
            color = (255, 0, 0)  # Red for Male
        else:
            female_count += 1
            label = "Female"
            color = (0, 255, 0)  # Green for Female

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 4)  
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA)

    print(f"Detected {male_count} Male(s) and {female_count} Female(s)")

    # Display the image with detections
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Gender Classification Output")
    plt.show()

    return male_count, female_count


# In[236]:


predict_group_image("Downloads/images/group image/7.jpg")


# In[237]:


predict_group_image("Downloads/images/group image/3.jpg")


# In[238]:


predict_group_image("Downloads/images/group image/4.jpg")


# In[239]:


predict_group_image("Downloads/images/group image/5.jpg")


# In[240]:


predict_group_image("Downloads/images/group image/6.jpg")


# In[241]:


predict_group_image("Downloads/images/group image/8.jpg")


# In[242]:


predict_group_image("Downloads/images/group image/9.jpg")


# In[243]:


predict_group_image(r"C:\Users\nikhi\OneDrive\Pictures\Camera Roll\WIN_20250402_15_45_08_Pro.jpg")


# In[2]:


import keras
print(keras.__version__)


# In[ ]:





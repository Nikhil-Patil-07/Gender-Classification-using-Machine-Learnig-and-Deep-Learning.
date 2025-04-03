import os
import dill
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# ✅ Set base directory
BASE_DIR = r"D:\computer vision\24070243038_Nikhil_Patil\Main_py"

# ✅ Function to check if a file exists
def check_file(filename):
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        print(f"⚠️ Error: File '{file_path}' not found. Make sure it is in the correct folder.")
        exit(1)
    return file_path

# ✅ Load models safely
cnn_model = load_model(check_file("gender_cnn_model.keras"), compile=False)
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

feature_extractor = load_model(check_file("feature_extractor.keras"))

# ✅ Load ML models
with open(check_file("svm_model.pkl"), "rb") as f:
    svm_model = pickle.load(f)

with open(check_file("rf_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(check_file("xgb_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

# ✅ Load the function with globals() to avoid AttributeError
with open(check_file("predict_function.pkl"), "rb") as f:
    loaded_function = dill.load(f, ignore=True)

# ✅ Continuous input loop for image predictions
while True:
    img_path = input("\n📌 Enter the image path (or type 'exit' to quit, without quotes): ").strip()

    if img_path.lower() == "exit":
        print("🚪 Exiting the program.")
        break  # Stop the loop

    if not os.path.exists(img_path):
        print(f"⚠️ Error: Image file '{img_path}' not found. Check the path and try again.")
        continue  # Ask for another input

    try:
        result = loaded_function(img_path)
        print(f"✅ Prediction Result: {result}")
    except Exception as e:
        print(f"❌ Error processing image: {e}")

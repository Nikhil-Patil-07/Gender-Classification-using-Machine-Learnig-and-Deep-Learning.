#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dill
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle


# In[2]:


cnn_model = load_model("gender_cnn_model.keras", compile=False)
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[3]:


import pickle
import cv2 

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)


# In[4]:


feature_extractor = load_model("feature_extractor.keras")


# In[5]:


import dill

# Load the function
with open("predict_function.pkl", "rb") as f:
    loaded_function = dill.load(f)


# In[6]:


loaded_function('Downloads/images/male/men1.jpg')


# In[ ]:





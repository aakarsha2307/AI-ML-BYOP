import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================================
# PAGE TITLE
# ======================================
st.set_page_config(page_title="Social Media Engagement Predictor", layout="centered")

st.title("📱 Social Media Engagement Prediction System")
st.write("Predict whether a social media post will have **Low, Medium, or High Engagement**.")

# ======================================
# CREATE DATASET MANUALLY
# ======================================
data = {
    "post_type": ["Reel", "Image", "Video", "Carousel", "Reel", "Image", "Video", "Carousel", "Reel", "Image"],
    "post_time": ["Morning", "Afternoon", "Evening", "Night", "Morning", "Evening", "Night", "Afternoon", "Morning", "Evening"],
    "likes": [1200, 800, 1500, 900, 1800, 700, 1300, 1000, 2000, 850],
    "comments": [150, 80, 200, 90, 250, 60, 180, 100, 300, 75],
    "shares": [300, 120, 400, 150, 500, 100, 350, 180, 600, 130],
    "engagement_level": ["High", "Medium", "High", "Medium", "High", "Low", "High", "Medium", "High", "Low"]
}

df = pd.DataFrame(data)

# ======================================
# ENCODE DATA
# ======================================
le_post_type = LabelEncoder()
le_post_time = LabelEncoder()
le_target = LabelEncoder()

df["post_type"] = le_post_type.fit_transform(df["post_type"])
df["post_time"] = le_post_time.fit_transform(df["post_time"])
df["engagement_level"] = le_target.fit_transform(df["engagement_level"])

# Features and target
X = df[["post_type", "post_time", "likes", "comments", "shares"]]
y = df["engagement_level"]

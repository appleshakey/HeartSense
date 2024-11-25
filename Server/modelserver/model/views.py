from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.views import Response
from rest_framework import status
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import pandas as pd
import numpy as np
import json
import joblib
import os


#initial code to load model

# Load and preprocess the data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    # Rename 'sex' to 'male' if needed
    if 'sex' in df.columns:
        df = df.rename(columns={'sex': 'male'})

    # Define features and target
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']

    # Store feature names
    feature_names = X.columns.tolist()

    return X, y, feature_names

# Create and train model with SMOTE
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to DataFrame to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Train Random Forest with balanced data
    rf_model = RandomForestClassifier(n_estimators=100,
                                    max_depth=10,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    random_state=42,
                                    class_weight='balanced')

    rf_model.fit(X_train_balanced, y_train_balanced)

    # Evaluate model
    # y_pred = rf_model.predict(X_test_scaled)
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    return rf_model, scaler

# Function to make predictions on new data
def predict_heart_disease(model, scaler, input_data, feature_names):
    input_data = input_data.rename(columns={'sex': 'male'} if 'sex' in input_data.columns else {})
    input_data = input_data.reindex(columns=feature_names)
    input_scaled = scaler.transform(input_data)
    input_scaled = pd.DataFrame(input_scaled, columns=feature_names)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    return prediction, prediction_proba

# Create your views here.
filepath = os.path.join(settings.BASE_DIR, 'framingham.csv')
print("Retrieving data...")
X, y, feature_names = load_and_preprocess_data(filepath)
print("training data...")
model, scaler = train_model(X, y)
print("training complete!")


class HelloWorld(APIView):
    def get(self, request):
        return Response(json.dumps({"message": "Hello World!!"}), status=status.HTTP_200_OK)
    
class FraminghamModel(APIView):
    def post(self, request):
        age = int(request.data["age"])
        education = int(request.data["education"])
        male = int(request.data["sex"] == "male")
        currentSmoker = int(request.data["currentSmoker"])
        cigsPerDay = int(request.data["cigsPerDay"])
        BPMeds = int(request.data["BPMeds"])
        prevalentStroke = int(request.data["prevalentStroke"])
        prevalentHyp = int(request.data["prevalentHyp"])
        diabetes = int(request.data["diabetes"])
        totChol = int(request.data["totChol"])
        sysBP = int(request.data["sysBP"])
        diaBP = int(request.data["diaBP"])
        BMI = int(request.data["BMI"])
        heartRate = int(request.data["heartRate"])
        glucose = int(request.data["glucose"])

        input = pd.DataFrame({
        'age': [age],
        'education': [education],
        'male': [male],  # Changed from 'sex' to 'male'
        'currentSmoker': [currentSmoker],
        'cigsPerDay': [cigsPerDay],
        'BPMeds': [BPMeds],
        'prevalentStroke': [prevalentStroke],
        'prevalentHyp': [prevalentHyp],
        'diabetes': [diabetes],
        'totChol': [totChol],
        'sysBP': [sysBP],
        'diaBP': [diaBP],
        'BMI': [BMI],
        'heartRate': [heartRate],
        'glucose': [glucose]
        })
        
        prediction, probability = predict_heart_disease(model, scaler, input, feature_names)
        # print("Risk Category:", "High Risk" if prediction[0] == 1 else "Low Risk")
        # print("Probability of High Risk: {:.2f}%".format(probability[0][1] * 100))
        return Response(json.dumps({'risk_category': "High Risk" if prediction[0] == 1 else "Low Risk", "probability": "{:.2f}%".format(probability[0][1]*100)}))


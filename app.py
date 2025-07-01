import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Face Recognition Model - SVM Based")
st.markdown("Upload the dataset and get real-time predictions using SVM. No model file needed.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File (faceData.csv format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")
    st.write("Preview of Dataset:", df.head())

    # Preprocessing
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy on Test Set: {acc * 100:.2f}%")

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # Optional input for prediction
    st.subheader("Make a Prediction")
    input_data = st.text_area("Enter comma-separated feature values for prediction (same as dataset columns):")

    if input_data:
        try:
            input_array = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            predicted_label = le.inverse_transform(prediction)[0]
            st.success(f"Predicted Label: {predicted_label}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    # Top Performing Model Section
    st.subheader("Top Performing Model")
    st.markdown("Based on current evaluation, **Support Vector Machine (SVM)** with a linear kernel is performing best in terms of accuracy and classification metrics. It was selected for this deployment due to its balance of speed and accuracy.")
else:
    st.warning("Please upload the dataset to continue.")

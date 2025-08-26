import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit layout
st.set_page_config(page_title="Cuisine Classifier", layout="wide")

st.title("🍽️ Cuisine Classification from Restaurant Data")

# --- Sidebar: Upload Dataset ---
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- Error handling for file upload ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data uploaded successfully!")

        # Show basic info
        if st.checkbox("Preview Dataset"):
            st.write(df.head())

        # --- Data Preprocessing ---
        st.header("🔄 Data Preprocessing")
        df.fillna("Unknown", inplace=True)

        # Encode cuisine (target variable)
        if 'Cuisine' not in df.columns:
            st.error("❌ 'Cuisine' column not found in dataset.")
            st.stop()

        label_encoder = LabelEncoder()
        df['Cuisine'] = label_encoder.fit_transform(df['Cuisine'])

        # One-hot encode other categorical features
        categorical_cols = df.select_dtypes(include='object').columns.drop('Cuisine', errors='ignore')
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df.drop("Cuisine", axis=1)
        y = df["Cuisine"]

        # Train-test split
        test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # --- Model Training ---
        st.header("🤖 Model Training")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Evaluation Metrics ---
        st.subheader("📊 Evaluation Metrics")
        report = classification_report(y_test, y_pred, output_dict=True, target_names=label_encoder.classes_)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=2))

        # --- Confusion Matrix ---
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # --- Feature Importance ---
        st.subheader("🌟 Top Features")
        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        st.dataframe(feature_df)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("📤 Please upload a dataset to begin.")

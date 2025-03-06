import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Machine Learning Model", page_icon="🤖")

st.title("Machine Learning Model Demo")

# Sample data generation
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 300
    features = np.random.randn(n_samples, 4)
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    return features, labels

# Model training function
@st.cache_resource
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Main app
def main():
    # Generate and split data
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_random_forest(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Display results
    st.header("Random Forest Classifier")
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    
    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': [f'Feature {i+1}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    st.bar_chart(feature_importance.set_index('feature'))

if __name__ == "__main__":
    main()
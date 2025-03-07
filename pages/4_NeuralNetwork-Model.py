import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page title and description
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")

st.title("Pneumonia Detection from X-ray Images")
st.write("Upload a chest X-ray image to check for pneumonia.")

# Function to load the model
@st.cache_resource
def load_pneumonia_model():
    try:
        model = load_model('./data/neural_network/pneumonia_model.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(img):
    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize the image to 224x224 for VGG16 input
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    
    return img_array

# Function to make predictions
def predict_pneumonia(model, img_array):
    prediction = model.predict(img_array)
    # If model returns multiple outputs, take the first one
    if isinstance(prediction, list):
        prediction = prediction[0]
    # Handle different output shapes
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        # For multi-class output
        return prediction[0]
    else:
        # For binary output
        return prediction[0][0]

# Main app functionality
def main():
    # Load the model
    model = load_pneumonia_model()
    
    if model is None:
        st.warning("Please make sure the model file exists at ./data/neural_network/pneumonia_model.h5")
        return
    
    # File uploader for X-ray images
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_bytes))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Uploaded X-ray Image", use_container_width=True)
        
        with col2:
            # Process the image and make prediction when user clicks the button
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Preprocess the image
                    processed_img = preprocess_image(img)
                    
                    # Show the preprocessed image for debugging
                    st.write("Preprocessed image shape:", processed_img.shape)
                    
                    # Make prediction
                    try:
                        prediction_result = predict_pneumonia(model, processed_img)
                        
                        # Handle different types of model outputs
                        if isinstance(prediction_result, np.ndarray) and len(prediction_result) > 1:
                            # For multi-class classification
                            class_names = ["Normal", "Pneumonia"]
                            predicted_class = np.argmax(prediction_result)
                            confidence = prediction_result[predicted_class] * 100
                            
                            if predicted_class == 1:  # Pneumonia
                                st.error(f"Pneumonia Detected (Confidence: {confidence:.2f}%)")
                            else:  # Normal
                                st.success(f"No Pneumonia Detected (Confidence: {confidence:.2f}%)")
                                
                            # Display all class probabilities
                            st.write("### Detailed Results:")
                            for i, class_name in enumerate(class_names):
                                st.write(f"{class_name}: {prediction_result[i] * 100:.2f}%")
                        else:
                            # For binary classification
                            if prediction_result > 0.5:
                                pneumonia_probability = prediction_result * 100
                                st.error(f"Pneumonia Detected (Confidence: {pneumonia_probability:.2f}%)")
                            else:
                                normal_probability = (1 - prediction_result) * 100
                                st.success(f"No Pneumonia Detected (Confidence: {normal_probability:.2f}%)")
                            
                            # Display prediction scores
                            st.write("### Detailed Results:")
                            st.write(f"Normal: {(1 - prediction_result) * 100:.2f}%")
                            st.write(f"Pneumonia: {prediction_result * 100:.2f}%")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.info("Try to check the model architecture and ensure the preprocessing matches the expected input.")
                    
                    st.warning("Note: This is an AI-based analysis and should not replace professional medical diagnosis.")

    # Information section
    st.sidebar.title("About")
    st.sidebar.info("""
    This application uses a deep learning model to detect pneumonia from chest X-ray images.
    
    **How to use:**
    1. Upload a chest X-ray image
    2. Click 'Analyze Image'
    3. View the results and probability scores
    
    **Important:** This tool is for demonstration purposes only and should not be used for medical diagnosis.
    """)

if __name__ == "__main__":
    main()
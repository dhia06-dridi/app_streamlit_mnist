import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Path to the model
model_path = "https://github.com/dhia06-dridi/app_streamlit_mnist/raw/main/mnist_cnn_model.keras"

# Loading the model
try:
    cnn_model = load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Dimensions for canvas and images
CANVAS_WIDTH = 200
CANVAS_HEIGHT = 200
IMG_SIZE = 28  # Target size for the model (28x28)

# Function to predict the image
def predict_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize between 0 and 1
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    predictions = cnn_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Streamlit application
st.title("Draw a Digit")

# Using st_canvas to draw
canvas_result = st_canvas(
    fill_color="black",  # Fill color of the background
    stroke_width=10,     # Brush width
    stroke_color="white", # Brush color
    background_color="black",  # Background color
    height=CANVAS_HEIGHT,
    width=CANVAS_WIDTH,
    drawing_mode="freedraw",  # Free drawing mode
    key="canvas",
)

# Check if the user has drawn
if canvas_result.image_data is not None:
    # Convert drawing to a PIL image
    image_array = (canvas_result.image_data[:, :, 0]).astype("uint8")  # Keep the image as is
    image = Image.fromarray(image_array)

    # Display the drawn image
    st.image(image, caption="Drawn Image", use_column_width=True)

    # Predict if the button is clicked
    if st.button("Predict"):
        try:
            predicted_class, confidence = predict_image(image)
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

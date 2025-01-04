import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Path to the model
model_path = "mnist_cnn_model.keras"

# Loading the model
try:
    cnn_model = load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Dimensions for images
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

# Drawing with Matplotlib
fig, ax = plt.subplots(figsize=(4, 4))
canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

# Display the blank canvas
im = ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
ax.set_axis_off()

# Function to draw on the canvas
def draw(event):
    global canvas, im
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            canvas[y, x] = 255
            im.set_data(canvas)
            fig.canvas.draw()

# Button for clearing the canvas
def clear_canvas(event):
    global canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    im.set_data(canvas)
    fig.canvas.draw()

clear_button = Button(plt.axes([0.8, 0.01, 0.1, 0.05]), 'Clear')
clear_button.on_clicked(clear_canvas)

# Connect the draw function to mouse click event
fig.canvas.mpl_connect('button_press_event', draw)

# Display the canvas
st.pyplot(fig)

# Predict if the button is clicked
if st.button("Predict"):
    try:
        image = Image.fromarray(canvas)
        predicted_class, confidence = predict_image(image)
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

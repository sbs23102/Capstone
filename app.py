import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
import cv2

# Define the dimensions of your input images
img_width, img_height = 224, 224

# Load the pre-trained model for food classification
food_model = tf.keras.models.load_model('capstone_labelled_food_classification_model3.h5', compile=False)
food_model.build((None, img_width, img_height, 3))

# Load the eco-score models
eco_score_models = {
    'en_biscuits': tf.keras.models.load_model('best_model_ecoscore_prediction_biscuits.h5', compile=False),
    'en_candies': tf.keras.models.load_model('best_model_ecoscore_prediction_candies.h5', compile=False),
    'en_cheeses': tf.keras.models.load_model('best_model_ecoscore_prediction_cheeses.h5', compile=False)
}

def preprocess_image(image, target_size=(img_width, img_height)):
    if isinstance(image, str):  # Check if it's a file path
        # Load the image using tf.keras.preprocessing.image
        img = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
    else:  # Assume it's a PIL Image object
        img = Image.fromarray(image)  # Convert NumPy array to PIL Image
        img = img.resize(target_size)

    # Convert the image to a NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Expand dimensions to create a batch (1, target_size[0], target_size[1], 3)
    img_array = tf.expand_dims(img_array, axis=0)

    # Preprocess the image for the food classification model
    img_array = mobilenet_preprocess_input(img_array)

    return img_array

def predict_food_category(img):
    # Make predictions on the image for food classification
    prediction = food_model.predict(img)
    class_names = ['en_biscuits', 'en_cheeses', 'en_candies']
    return class_names[np.argmax(prediction)]

def remove_background(image):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Extract the green color channel (assuming the image is in BGR format)
    green_channel = img_array[:, :, 1]  # Green channel is at index 1

    # Apply thresholding to create a binary mask based on the green channel
    _, mask = cv2.threshold(green_channel, 254, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img_array, img_array, mask=mask_inv)

    return result

def preprocess_image_eco_score(image, target_size=(img_width, img_height)):
    # Additional preprocessing specific to the eco-score models, if needed
    # ...

    return preprocess_image(image, target_size)  # Reuse the main preprocessing function for simplicity

def predict_eco_score(img, category):
    # Use the appropriate eco-score model based on the predicted category
    eco_score_model = eco_score_models.get(category)
    if eco_score_model is None:
        return None  # Handle the case where the eco-score model is not found

    # Make predictions on the image for eco-score
    eco_score_prediction = eco_score_model.predict(img)
    return eco_score_prediction

def main():
    st.title("Food Image Classifier with Eco-Score")

    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Remove background
        img_no_background = remove_background(image)
        st.image(img_no_background, caption="Image without Background", use_column_width=True)

        # Preprocess and make predictions for food category
        img_food = preprocess_image(img_no_background)
        food_category = predict_food_category(img_food)

        # Display the food category prediction
        st.subheader("Food Category Prediction:")
        st.write(food_category)

        # Preprocess image for eco-score model
        img_eco_score = preprocess_image_eco_score(img_no_background)

        # Make predictions for eco-score based on the food category
        eco_score_prediction = predict_eco_score(img_eco_score, food_category)

        # Display the eco-score prediction
        st.subheader("Eco-Score Prediction:")
        if eco_score_prediction is not None:
            st.write(eco_score_prediction)

if __name__ == "__main__":
    main()

import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
import numpy as np

# Define the dimensions of your input images
img_width, img_height = 150, 150

# Load the pre-trained models
food_classification_model_1 = tf.keras.models.load_model('food_classification_model_Pre_2.h5', compile=False)
food_classification_model_1.build((None, img_width, img_height, 3))

food_classification_model_2 = tf.keras.models.load_model('food_classification_model_Pre.h5', compile=False)
food_classification_model_2.build((None, img_width, img_height, 3))

eco_score_model = tf.keras.models.load_model('model_ecoscore_score_score_prediction_allcats_app.h5', compile=False)
eco_score_model.build((None, img_width, img_height, 3))

nutri_score_model = tf.keras.models.load_model('model_nutriscore_score_prediction_allcats_app.h5', compile=False)
nutri_score_model.build((None, img_width, img_height, 3))

def preprocess_image(image, target_size=(img_width, img_height)):
    if isinstance(image, str):  # Check if it's a file path
        # Load the image using tf.keras.preprocessing.image
        img = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
    else:  # Assume it's a PIL Image object
        img = image.resize(target_size)

    # Convert the image to a NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Expand dimensions to create a batch (1, target_size[0], target_size[1], 3)
    img_array = tf.expand_dims(img_array, axis=0)

    # Preprocess the image for the food classification model
    img_food_classification = mobilenet_preprocess_input(img_array)

    return img_food_classification

def get_eco_category(score):
    # Ensure the score is within the valid range [0, 100]
    score = max(0, min(score, 100))

    if 80 <= score <= 100:
        return "A", "darkgreen"
    elif 60 <= score < 80:
        return "B", "lightgreen"
    elif 40 <= score < 60:
        return "C", "yellow"
    elif 20 <= score < 40:
        return "D", "orange"
    else:
        return "E", "red"

def get_nutri_category(score):
    # Ensure the score is within the valid range [0, 100]
    score = max(0, min(score, 100))
    
    if score <= -1:
        return "A", "darkgreen"
    elif 0 <= score <= 2:
        return "B", "lightgreen"
    elif 3 <= score <= 10:
        return "C", "yellow"
    elif 11 <= score <= 18:
        return "D", "orange"
    else:
        return "E", "red"

def main():
    st.title("Food Image Classifier with Eco-Score and Nutri-Score")

    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess and make predictions for food category (model 1)
        img_food_classification_1 = preprocess_image(image)
        food_classification_prediction_1 = food_classification_model_1.predict(img_food_classification_1)
        food_class_names_1 = ['en_candies', 'en_cheeses']
        food_classification_category_1 = food_class_names_1[np.argmax(food_classification_prediction_1)]
        food_classification_confidence_1 = np.max(food_classification_prediction_1)

        # Preprocess and make predictions for food category (model 2)
        img_food_classification_2 = preprocess_image(image)
        food_classification_prediction_2 = food_classification_model_2.predict(img_food_classification_2)
        food_class_names_2 = ['en_biscuits', 'en_cheeses']
        food_classification_category_2 = food_class_names_2[np.argmax(food_classification_prediction_2)]
        food_classification_confidence_2 = np.max(food_classification_prediction_2)

        # Display the food category prediction with the higher confidence score
        if food_classification_confidence_1 > food_classification_confidence_2:
            st.subheader("Food Category Prediction:")
            st.write(food_classification_category_1)
        else:
            st.subheader("Food Category Prediction:")
            st.write(food_classification_category_2)

        # Add description for Eco-Score model
        st.subheader("Eco-Score Model:")
        st.write(
            "Public data : quantitative data on product Life Cycle Assessment (LCA) from the Agribalyse database, drawn up by experts and implemented in Agribalyse. Impacts on environment through production, transport, fabrication, and packaging are taken into account, giving a score out of 100. Data not included in LCA but which do take into account the positive or negative impact on the environment: data on the product label or given by the producer, as well as additional quality criteria : recyclability of packages, labels (Bio, quality etc.), where the ingredients come from, seasonality of food used (for recipes and ready meals). All these data will give a bonus/malus which will influence the score. The total mark out of 100 gives a score from A to E."
        )

        # Make predictions for Eco-Score
        eco_score_prediction = eco_score_model.predict(preprocess_image(image))

        # Get Eco-Score category and color
        eco_category, eco_color = get_eco_category(eco_score_prediction[0][0])

        # Display the Eco-Score prediction and category
        st.subheader("Eco-Score Prediction:")
        st.write(eco_score_prediction[0][0])
        st.subheader("Eco-Score Category:")
        st.write(f'<p style="color:{eco_color}; font-size: 36px;">{eco_category}</p>', unsafe_allow_html=True)

        # Make predictions for Nutri-Score
        nutri_score_prediction = nutri_score_model.predict(preprocess_image(image))

        # Get Nutri-Score category and color
        nutri_category, nutri_color = get_nutri_category(nutri_score_prediction[0][0])

        # Add description for Nutri-Score model
        st.subheader("Nutri-Score Model:")
        st.write(
            "A Nutri-Score for a particular food item is given in one of five classification letters, with 'A' being a preferable score and 'E' being a detrimental score. Products with a NutriScore value of -1 or below receive an A grade, while those with a value between 0 and 2 are classified as B. Products scoring between 3 and 10 receive a C grade, whereas those scoring 11 to 18 are assigned a D grade. Finally, products with a NutriScore value above 19 receive an E grade."
        )

        # Display the Nutri-Score prediction and category
        st.subheader("Nutri-Score Prediction:")
        st.write(nutri_score_prediction[0][0])
        st.subheader("Nutri-Score Category:")
        st.write(f'<p style="color:{nutri_color}; font-size: 36px;">{nutri_category}</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

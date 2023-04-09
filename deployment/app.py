import streamlit as st
import PIL
import requests, io
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os

st.header('Fish Image Prediction')
st.write("""
Created by Prayoga Agusto Haradi.
""")

st.subheader('Explanation & Context of This Prediction Model')
st.write("""
Currently, this model is only able to accurately predict 9 fish types.
Those fish types consists of: Trout, Black Sea Sprat, Sea Bass, Red Mullet,
Horse Mackerel, Gilt-Head Bream, Red Sea Bream, Striped Red Mullet, and Shrimp.
""")
st.write("""
""")
st.write("""
If you believe the fish image you wish to predict belong to any of these classes,
then the model will try and predict the image of the fish you uploaded into any of those 9 classes.
But be aware that the model is not perfect, due to the fact that this model has only been trained
on a low amount of images. Model improvements will come in the future.
""")
st.subheader('Example Images That You Can Use')
st.write("""
You can use the example images of the 9 classes in my [dropbox](https://www.dropbox.com/sh/70b4f8i5un9sw36/AACRElVoJ43E7TDY-c49rfMJa/For%20Deployment?dl=0&subfolder_nav_tracking=1) if you wish to test it first.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# Check if image is uploaded
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Image.', use_column_width=True)


def predict(model, pic):
    '''
    Fungsi untuk memprediksi kelas ikan
    '''
    labels = {'Black Sea Sprat': 0, 'Gilt-Head Bream': 1,
              'Hourse Mackerel': 2, 'Red Mullet': 3,
              'Red Sea Bream': 4, 'Sea Bass': 5,
              'Shrimp': 6, 'Striped Red Mullet': 7, 'Trout': 8}
    labels = dict((v, k) for k, v in labels.items())
    img = tf.keras.preprocessing.image.load_img(pic, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    # Use the model to predict the class probabilities for the image
    predictions = model.predict(img)
    # Get the index of the class with highest probability
    pred_idx = np.argmax(predictions, axis=1)[0]
    pred_prob = predictions[0][pred_idx]
    # Get the corresponding class label
    prediction = labels[pred_idx]
    return prediction, pred_prob


# Load the model
model = tf.keras.models.load_model('model.h5')

# Call the predict function with the loaded model
if st.button('Predict'):
    if uploaded_file is not None:
        filename = uploaded_file.name
        image = PIL.Image.open(uploaded_file)
        img_bytes = io.BytesIO()
        if os.path.splitext(filename)[1].replace(".", "").lower() == 'jpg':
            image.convert('RGB').save(img_bytes, format='JPEG')
            image.save(img_bytes, format='JPEG')
        else:
            image.save(img_bytes, format=os.path.splitext(filename)[1].replace(".", "").lower())
        img_bytes.seek(0)
        prediction, pred_prob = predict(model, img_bytes)
        st.write("The predicted fish species is:")
        st.write(prediction)
        st.write("Probability:")
        st.write(round(pred_prob * 100,2),'%')
        prediction_made = True
    else:
        st.write("Please upload an image to make a prediction.")
        prediction_made = True
else:
    st.write("Please upload an image and press the predict button to predict your fish type.")

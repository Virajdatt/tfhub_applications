import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
import numpy as np
import streamlit as st

IMAGE_SHAPE = (224, 224)

mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
classifier = tf.keras.Sequential([
    hub.KerasLayer(mobilenet_v2, input_shape=IMAGE_SHAPE+(3,))
])

print('HI')

@st.cache()
def preprocess_inference_image(img: str):
    img = Image.open(img).resize(IMAGE_SHAPE)
    img = np.array(img)/255.0
    #img = preprocess_inference_image('../sample_image.png')
    print('Old shape', img.shape)
    img_expanded = img[np.newaxis, ...]
    print('New shape', img_expanded.shape)
    return img_expanded

#image_expanded = preprocess_inference_image()

@st.cache()
def predict_results(image_expanded):
    result = classifier.predict(image_expanded)
    predicted_class = tf.math.argmax(result[0], axis=-1)
    labels_path = '/Users/virajdatt/Desktop/github/public/tfhub_applications/ImageNetLabels.txt'
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    predicted_class_name = imagenet_labels[predicted_class]
    return predicted_class_name


### Excluding Imports ###
st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    image_expanded = preprocess_inference_image(uploaded_file)
    result_class   = predict_results(image_expanded)
    
    st.write(f'The image is recognized as', result_class)


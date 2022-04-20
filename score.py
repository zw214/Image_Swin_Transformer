
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Mammography Classifier')
st.text('Provide URL of Mammography image')

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('./models')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['negative', 'benign calcification', 'benign mass', 
         'malignant calcification', 'malignant mass']

def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return tf.image.resize(image, [128, 128])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter image URL to Classify.. ', 'http://undergradimaging.pressbooks.com/wp-content/uploads/sites/66470/2017/10/Mammography-images-a-300x244.png')
if path is not None:
  content = requests.get(path).content

  st.write('Predicted Class : ')
  with st.spinner('classifying...'):
    label = np.argmax(model.predict(decode_img(content)), axis=1)
    st.write(classes[label[0]])
  st.write("")
  image = Image.open(BytesIO(content))
  st.image(image, caption='Classifying Mammo Image', use_column_width=True)

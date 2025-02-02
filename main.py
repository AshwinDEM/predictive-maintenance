import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import logging

logging.basicConfig(level=logging.ERROR)
st.title("Engine Maintenance")

file = st.file_uploader("Enter your data file")

binary_model = load_model("models/model.keras")

rul_model = load_model("models/model.keras")

sequence_length = 50
num_features = 24

if file is not None:
    df = pd.read_csv(file)
    st.write(df)
    data= df.to_numpy().reshape(1, sequence_length, num_features)

    prediction = binary_model.predict(data)
    prediction = prediction[0][0]

    col1, col2 = st.columns(2)
    with col1:
        if round(prediction):
            st.title("The engine is not damaged")
        else:
            st.title("The engine is damaged")
    with col2:
        RUL = rul_model.predict(data)
        st.title(f"The number of cycles left = {round(1 / RUL[0][0])}")
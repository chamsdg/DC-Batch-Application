# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:36:59 2020

@author: Chamsedine
"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import io

model = load_model('deployment_dc')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo_train-data.png')
    #image_hospital = Image.open('hospital.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to resolve the classification probleme')
    st.sidebar.success('https://www.linkedin.com/in/chamsedineaidara')
    
    #st.sidebar.image(image_hospital)

    st.title("Classification App")


    if add_selectbox == 'Batch':
        st.set_option('deprecation.showfileUploaderEncoding', False)

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)#to have the possibilities to download a csv file
            
            

if __name__ == '__main__':
    run()

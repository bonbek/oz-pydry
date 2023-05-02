import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Modeling"
sidebar_name = "Modeling"


def run():

    st.title(title)

    st.markdown(
        """
        Score par climat d'une Regression Logistique
        """
    )
    st.image(Image.open("assets/resampling-by-climate.png"))




    
    st.markdown(
        """
        model KNN, RF, SVM, GridSearch
        """
    )

    st.markdown(
        """
        Résulats des sg modèles par climats
        """
    )

    st.image(Image.open("assets/accuracy-by-estimator.png"))
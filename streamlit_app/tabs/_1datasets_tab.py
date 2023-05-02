import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Datasets"
sidebar_name = "Datasets"


def run():

    st.title(title)

    st.subheader(
        """
       Présentation du jeu de donnée et volumétrie 
       """
       )
    
    st.markdown(
        """
    Le jeu de données Rain in Australia étudié dans ce rapport provient de Kaggle (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package). 
       Ce jeu rassemble 10 années d'observations quotidiennes provenant de ~8000 stations météorologiques réparties sur le territoire australien pour la période du 01/11/2007 au 25/06/2017. 
       Des données supplémentaires (observations, statistiques...) sont accessibles aux services de données du Bureau Météorologique Gouvernemental Australien.

       
       La composition du dataset est de 145 460 observations, accompagné de 22 variables explicatives. Dont quatre variables ayant plus de 40% de NaNs
       
       """
    )
    st.image(Image.open("assets/variables-datasets.png"), caption = "Tableau des variables du jeu de données")

    st.markdown(
        """
        Concernant la répartition des observations au sein des zones climatiques, nous trouvons peu de donnée sur les zones climatique Aride et Tropical

        """
     )
    st.image(Image.open("assets/sample-count-by-climate.png"), caption = "Volumétrie par zone climatique")

    st.markdown(
        """
        A propos de la varaible cible "RainTomorow", les données de précipitation "Yes", sont très peu fréquentes dans les zones climatique aride.
        Au global nous constatons une classe "Yes" déséquilibrée par rapport à la classe "No".
        """
    )

    st.image(Image.open("assets/sample-class-by-climate.png"), caption = "Répartition de la variable cible")

    st.markdown(
        """
        La matrice de correlation nous indique quelques variables très corrélées avec la variable cible.
        Nous apercevons également des varibles explicative corrélées entres-elles.

        """
    )
    st.image(Image.open("assets/Correlation-tab.png"), caption ="Matrice de corrélation")

    
 
run


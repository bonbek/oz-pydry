import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import utils.layout as layout

layout.base_styles()
layout.show_members()


st.title(
    """
    Exploration des données
    """
)

st.markdown("---")

st.markdown(
    """
    Le jeu de données Rain in Australia étudié dans ce projet provient de [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package). 
    Ce jeu rassemble 10 années d'observations quotidiennes provenant de ~8000 stations météorologiques réparties sur le territoire australien pour la période du 01/11/2007 au 25/06/2017.
    """
)

st.markdown("---")

st.header(
    """
    Volumétrie
    """
)

st.markdown(
    """
    La composition du dataset est de __145 460__ observations, accompagné de __22__ variables explicatives. Dont quatre variables ayant plus de :orange[40% de NaNs]
    """
)

st.image(Image.open("assets/variables-datasets.png"), caption = "Descriptif des variables du jeu de données")

st.subheader(
    """
    Répartition des observations
    """
)

st.markdown(
    """
    Concernant la répartition des observations au sein des zones climatiques, nous trouvons peu de données pour les zones Aride et Tropicale.
    """
)

st.image(Image.open("assets/sample-count-by-climate.png"), caption = "Répartition des observations par zones climatiques")

st.markdown(
    """
    A propos de la variable cible :blue[RainTomorow], les données de précipitation _Yes_, sont très peu fréquentes dans les zones climatiques arides.
    Au global nous constatons une :orange[classe _Yes_ déséquilibrée] par rapport à la classe _No_.
    """
)

st.image(Image.open("assets/sample-class-by-climate.png"), caption = "Répartition de la variable cible")

st.markdown(
    """
    La matrice de correlation nous indique quelques variables très corrélées avec la variable cible.
    Nous apercevons également des variables explicatives corrélées entres-elles.
    """
)
st.image(Image.open("assets/Correlation-tab.png"), caption ="Matrice de corrélation")


class_names = ["No_Rain", "Rain"]


title_1 = "Identification valeurs aberrantes et des données incohérentes"
title_2 = "Distribution des caractéristiques"
sidebar_name = "Exploration des données"


# TODO: choose between one of these GIFs
# st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
#st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
# st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

st.markdown("---")

st.subheader(title_1)

st.markdown(
    """
Les valeurs aberrantes sont des points de données extrêmes qui se situent au-delà des normes attendues pour leur type (i.e. Variable), et sachant que  ces données aberrantes peuvent entraver les spécifications du modèle, déconcerter l'estimation des paramètres et générer des informations incorrectes, alors, il demeure primordial de les détecter et de les corriger afin d’éviter les problèmes déjà signalés.

Une façon assez simple de détecter ces valeurs est de réaliser un box-plot pour chacune des variables. 

    """
        )
st.image("assets/a.png")


st.markdown(
    """
    Une première figure qui rassemble les box-plot de l’ensemble des variables quantitatives montrent clairement l'existence des valeurs aberrantes surtout pour les variables Rainfall et Evapration.
    """
)

st.image("assets/b.png")

st.markdown(
    """
    Pour les variables de type températures, on remarque, que les valeurs aberrantes sont moins présentes, et on peut même les considérer comme des valeurs extrêmes.

    """
)



st.image("assets/c.png")

st.markdown(
    """
        Pour des variables comme cloud , sunshine, il y a une absence des valeurs aberrantes.

    """
)

st.image("assets/d.png")

st.header(title_2)

st.markdown(
    """
        Pour ce projet, on est devant deux types de variables (quantitative et catégorielle), une présentation graphique 
        de la distribution sera utile pour avoir une idée sur la nature de la distribution et aussi sur la grandeur numérique 
        de chaque variable.


    """
)

st.subheader("Variables quantitatives")

st.image("assets/e.png")

st.markdown(
    """

On remarque clairement la nécessité de normalisation des valeurs vu la diversité des échelles au niveau 
des valeurs de chaque distribution. 

    """
)

st.subheader("Variables catégorielles")

st.image("assets/f.png")

st.markdown(
    """



    """
)





st.markdown(
    """



    """
)


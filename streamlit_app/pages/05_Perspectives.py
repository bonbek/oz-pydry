import streamlit as st
import pandas as pd
import numpy as np
import utils.layout as layout
import os
import sys
sys.path.insert(0, os.path.abspath("../lib"))
from ozpydry.metrics import ClfReport

layout.base_styles()
layout.show_members()

st.title(
    """
    Autocritique et perspectives
    """)

st.image('./assets/comic-strip.png')

st.subheader(
    """
    Bilan
    """
)
st.markdown(
    """
    Après avoir étudié les caractéristiques de notre base de données, nous avons établi plusieurs hypothèses afin de créer un modèle baseline.
    Nous l'avons ensuite comparé avec d'autres modèles de machine learning tout en appliquant des preprocessing spécifiques: avec ou sans PCA, avec ou sans les valeurs manquantes, avec les 3 climats, les 9 sous-climats ou l’ensemble d'entraînement, l’équilibrage des classes...

    Les performances de ces modèles étant peu concluantes, nous avons donc appliqué un modèle de Deep Learning qui s'est révélé plus homogène dans ses résultats, notamment sur le _recall_ et la _precision_ de la classe minoritaire _(Yes)_.

    Nous sommes arrivés proches des résultats obtenus par les modèles de prédiction utilisés en Australie sur la période étudiée :smiley:.
    
    En fin de compte nous retirons une certaine déception sur les performances obtenues, alors que nous nous sommes focalisés sur une seule variable cible binaire.
    """
)

st.subheader(
    """
    Pistes d'amélioration
    """
)

st.markdown(
    """
    Avec du temps supplémentaire, nous aurions étudié plus en détail l'aspect temporel des variables, en effet, à mesure de notre avancement nous avons remarqué des "comportements saisonniers" sur certains de nos modèles (additionnés aux climats).
    """
)

with st.expander("Aperçu dans le temps de prédictions projetées sur les variables principales"):
    st.image('./assets/plot-preds-seasonal-features.png')
    report = ClfReport('./assets/per-season-climate-lr.json')
    st.write(report.to_markdown(compact=True, tone=True, title='Per season & climate model _(LogReg.)_'), unsafe_allow_html=True)
    st.write("""
    <style>
    .rt tr > td:first-child {
        width: 30%;
    }
    </style>
    """,unsafe_allow_html=True)

st.markdown(
    """
    Nous aurions aussi consolidé le dataset avec des nouvelles variables issue de la topographie des lieux, à l'image du modèle AROME (modèle Français). 
    
    Pour ces axes de recherche, des modèles de type RNN (ou du moins une combinaison avec) nous semblent une piste adaptée.
    """
)

st.subheader(
    """
    Rappel des enjeux
    """
)

st.markdown(
    """
    Le 18 Août 2022, une tempête en Corse a causé 5 décès, ainsi que de gros dégâts matériels. Pourtant, la veille, Météo France avait classé la Corse en vigilance jaune. 
    Bien que les modèles se soient améliorés ces dernières décennies, il y encore une grande marge pour s’approcher des 100% de précision mais aussi gagner dans la portée à J+14.

    Les conditions climatiques à venir vont avoir de lourdes conséquences pour l'ensemble de la population mondiale, les bonnes prédictions climatiques n’ont jamais été aussi importantes pour anticiper les catastrophes naturelles à venir.
    """
)
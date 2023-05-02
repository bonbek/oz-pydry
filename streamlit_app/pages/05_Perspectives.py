import streamlit as st
import pandas as pd
import numpy as np
import utils.layout as layout

layout.base_styles()
layout.show_members()

title = "Regard critique et perspectives"
sidebar_name = "Regard critique et perspectives"


st.title(title)

st.subheader(
    """
    Bilan
    """
)
st.markdown(
    """
    Après avoir étudié les caractéristiques de notre base de données lors de l’exploration, nous avons établi plusieurs hypothèses afin de créer le modèle le plus performant en précision. 
    Une fois notre premier modèle naïf créé nous avons comparé avec différents modèles de machine learning tout en
    appliquant des comparaisons selon un preprocessing spécifique (avec ou sans PCA, avec ou sans les lignes contenant des NaNs, Avec les 3 climats, les 9 sous-climats ou l’ensemble d'entraînement, les différents modes d’équilibrage des classes. 
    Mais les résultats de précision et de recall avec les modèles de machine learning n’ont pas été suffisamment concluants pour nous. Nous avons donc appliqué un modèle de Deep Learning, les résultats obtenus sont meilleurs que les modèles de machine learning,  notamment sur le recall de la classe 1. 
    On se rapproche des résultats des modèles de prédiction réels utilisés en Australie sur la période étudiée.
    """
)

st.subheader(
    """
    Amélioration et Perspectives
    """
)
st.markdown(
    """
    Avec du délais supplémentaire nous aurions utilisé les données temporelles pour réalisé un modèle RNN
    Nous aurions consolidé le dataset avec des nouvelles variables issue de la topographie des lieu, à l'image du modèle AROME (modèle Français)
    En fin de compte nous retenons une certaine déception sur les précisions obtenues qui ne sont pas meilleurs que les modèles météorologiques réel, alors que nous somme focalisé sur qu'une seule variable cible.
    
                
    Rappelons que le 18 Août 2022, une tempête en Corse a causé 5 décès, ainsi que de gros dégâts matériels. Pourtant, la veille, Météo France avait classé la Corse en vigilance jaune. 
    Bien que les modèles ont gagné en précisions sur les dernières décennies, il y encore beaucoup de marge à gagner pour s’approcher des 100% de précision mais aussi gagner dans la portée de prédiction à J+14.
    Les conditions climatiques à venir vont avoir de lourdes conséquences pour l'ensemble de la population mondiale, les bonnes prédictions climatiques n’ont jamais été aussi importantes pour anticiper les catastrophes naturelles à venir.
    
    """
)
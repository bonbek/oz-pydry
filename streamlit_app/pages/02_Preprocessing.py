import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0, os.path.abspath("../lib"))
from ozpydry.metrics import ClfReport
import utils.layout as layout

layout.base_styles()
layout.show_members()


title = "Preprocessing"
sidebar_name = "Preprocessing"


st.title(title)


st.markdown(
    """
    Concernant la gestion des NaNs, nous évaluons nos différents modèles sans les lignes contenant les NaNs, en les transformant par le mode ou encore la moyenne. 
    """
)
st.markdown(
    """
    Nous avons discrétisé les variables catégorielles pour qu'elles puissent être traitées par tous les modèles 
    """
)

st.markdown(
    """
    Nous avons eu l'idée de créer les modèles en fonction des 3 zones climatiques. La méthode de clustering n'as pas abouti à nos attentes de labellisation par localisation. Nous avons donc labelisé manuellement les lieux en fonction de leur situation géographique.
    """
)

st.markdown(
    """
    Pour l'équilibrage des données, nous évaluons systématiquement les deux méthodes de réquilibrage (over/under sampling) pour les modèles. La méthode under sampling reste souvent la plus performante.
    """
)

st.markdown(
    """
    Notre apporche pour optimiser nos calculs via la méthode pca n'a pas été concluant, chaque modèle perd en précision et le temps de calcul n'est pas plus rapide. Nous avons donc utilisé toutes les variables
    """
)
# st.image(Image.open("assets/variables-pca.png"))
report2 = ClfReport("assets/pca-unbalanced.json")

    
# st.markdown(show(report2, tone = True, compact = True, title= "**Score PCA**"), unsafe_allow_html= True)
with st.expander("Résultat PCA"):
    st.write("Voici les données :", report2.to_markdown(tone = True, compact = True, title= "**Score PCA**"), unsafe_allow_html= True)
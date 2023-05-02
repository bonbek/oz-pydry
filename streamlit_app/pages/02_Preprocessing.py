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


def show(report, include=None, compact=False, title=None, tone=False):
    if isinstance(include, str):
        include = [include]

    reps = [(k, report.reports_[k]) for k in include] \
                if hasattr(include, "__iter__") else report.reports_.items()

    return report.tomdc_(reps, tone, title) if compact else report.tomdf_(reps, tone, title)

title = "Preprocessing"
sidebar_name = "Preprocessing"


st.title(title)

st.markdown(
    """
    Gestion des Nans, PCA, création des 3 cluster Climats, équilibrage des données
    """
)

st.markdown(
    """
    Résulats des pca
    """
)

st.image(Image.open("assets/variables-pca.png"))
report1 = ClfReport("assets/balancing-report.json")
report2 = ClfReport("assets/pca-unbalanced.json")
report3 = ClfReport("assets/feature-sel-report.json")

st.markdown(report1.to_markdown(compact=True, tone=True, title= "**Score KNN**"), unsafe_allow_html=True)
# with st.expander("Résultat KNN"):
#     st.write("Voici les données :", st.markdown(show(report1, tone = True, compact = True, title= "**Score KNN**"), unsafe_allow_html= True))


st.markdown(report2.to_markdown(tone = True, compact = True, title= "**Score PCA**"), unsafe_allow_html= True)
with st.expander("Résultat PCA"):
    st.markdown(report2.to_markdown(tone = True, compact = True, title= "**Score PCA**"), unsafe_allow_html= True)


st.markdown(report3.to_markdown(tone = True, compact = True, title= "**Feature Sel Report**"), unsafe_allow_html= True)



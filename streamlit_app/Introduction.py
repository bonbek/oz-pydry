import streamlit as st
import utils.layout as layout

layout.base_styles()
layout.show_members()

title = "Gonna rain tomorow ?"
sidebar_name = "Introduction"


# TODO: choose between one of these GIFs
# st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
# st.image("https://thumbs.gfycat.com/TallDrearyBlueswhale-max-1mb.gif", )
st.image("https://thumbs.gfycat.com/DearestEntireHippopotamus.webp")

st.title(title)

st.markdown("---")
st.subheader("Présentation du projet et des enjeux:")


st.markdown(
    """
    
    "Quel temps fera-t-il demain ?" est peut être l'une des questions les plus fréquentes et répandue depuis que l'humanité projète des plans.
    Bien avant l'arrivée des systèmes informatiques, certains étudiaient les différentes conditions climatiques pour établir des prédictions.
    Depuis l'instalation de sondes et l'utilisations de systèmes informatiques, la précision de ces prédictions météorologique s'est sensiblement améliorée.
    
    L'Australie est un pays avec des zones arides et semi-arides importantes, les incendies de forêts y sont fréquents, surtout pendant les périodes de sécheresse. La pluie est vitale pour la survie de la faune et de la flore et est essentielle pour l'agriculture, l'un des piliers économiques de l'Australie.
    Bien que les précipitations soient souvent faibles, tout au long de son histoire, l'Australie à subit des inondations aux conséquences dévastatrices. La prévision de la pluie peut permettre de prendre des mesures pour minimiser les dommages causés par les inondations.
    En somme, la prévision de la pluie en Australie est cruciale pour la sécurité et le bien-être des êtres humains, de la faune, de la flore et de l'économie.
    Nous souhaitons donc contribuer à la précision des prévisions météorologiques en Australie.
    Pour cela, nous utiliserons leurs données recensées en Australie avec nos modèles de machine learning et de deep learning.
    
    """
)

st.subheader(
    """
    Les zones climatiques de l'Australie :
    """
)
st.image("https://www.gostudy.fr/wp-content/uploads/2019/07/10-Maps-of-Australia-FR_Climats.jpg", caption = "Zones climatiques en Australie")

st.markdown(
    """
    Il existe 3 grandes zones climatiques: Tropical, Aride et Tempéré.
    Chacune des zones ont des caractéristiques propres à leur climat. Le Nord, avec un climat tropical est composé de 2 saisons, "humide" et "sèche". 
    Au centre du pays le climat est aride, il y a deux saisons également, "chaude" et "fraiche" et la pluie quasi absente toute l'année.
    Au sud, la zone a 4 saisons: hiver, printemps, été, automne.   
    """
)
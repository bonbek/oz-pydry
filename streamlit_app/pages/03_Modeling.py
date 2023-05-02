import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

from sklearn import model_selection
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Sequential, load_model, Model

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

from joblib import load

import h5py

from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

import utils.layout as layout

layout.base_styles()
layout.show_members()

title = "Machine Learning VS Deep Learning"
sidebar_name = "Machine Learning VS Deep Learning"


st.title(title)


st.subheader("Choisir une zone géographique") # st.sidebar
classifier_z = st.selectbox(" ", ("--------------------","Zone A : Aride","Zone B : Tempérée","Zone C : Tropicale","Toutes les zones"))

if classifier_z == "--------------------":
    strz="nan"
if classifier_z == "Zone A : Aride":
    strz="A"
if classifier_z == "Zone B : Tempérée":
    strz="B"
if classifier_z == "Zone C : Tropicale":
    strz="C"
if classifier_z == "Toutes les zones":
    strz="All"

st.subheader("Choisir une méthode de rééchantillonnage ") # st.sidebar
classifier_e = st.selectbox(" ", ("--------------------","Sans","Sur-échantillonnage Oversampling","Sous-échantillonnage Undersampling"))

if classifier_e == "--------------------":
    stre="nan"
if classifier_e == "Sans":
    stre=""
if classifier_e == "Sur-échantillonnage Oversampling":
    stre="O"
if classifier_e == "Sous-échantillonnage Undersampling":
    stre="U"


if (strz!="nan") and (stre!="nan"):
    # charger les Modèles
    clf_KNN = load('./models/clf_KNN_zone_'+strz+stre+'.joblib')
    clf_SVM = load('./models/clf_SVM_zone_'+strz+stre+'.joblib')
    clf_RF = load('./models/clf_RF_zone_'+strz+stre+'.joblib')
    RND = load('./models/CNN_zone_'+strz+stre+'.joblib')


    # Charger split data
    hf =h5py.File('./models/data_train_test_zone_'+strz+stre+'.h5', 'r')
    n1 = hf.get('X_train_scaled')
    X_train_scaled = np.array(n1)
    n2 = hf.get('X_test_scaled')
    X_test_scaled = np.array(n2)
    n3 = hf.get('y_train')
    y_train = np.array(n3)
    n4 = hf.get('y_test')
    y_test = np.array(n4)

st.subheader("Choisir un modèle") # st.sidebar
classifier = st.selectbox("", ("--------------------","K-nearest neighbors (KNN)", "Support Vector Machine (SVM)", "Random Forest (RF)","Les réseaux de neurones (Dense)"))


if classifier == "--------------------":
    strm="nan"

if classifier == "K-nearest neighbors (KNN)":
    strm=""
    Model_used= clf_KNN
    y_pred_RF = Model_used.predict(X_test_scaled)
if classifier == "Support Vector Machine (SVM)":
    strm=""
    Model_used= clf_SVM
    y_pred_RF = Model_used.predict(X_test_scaled)
if classifier == "Random Forest (RF)":
    strm=""
    Model_used= clf_RF
    y_pred_RF = Model_used.predict(X_test_scaled)
if classifier == "Les réseaux de neurones (Dense)":
    strm=""
    Model_used= RND
    y_pred_RF = Model_used.predict(X_test_scaled)
    y_pred_RF = np.argmax(y_pred_RF,axis=1)


st.subheader("Résultats obtenus")

#st.markdown(""This text is :red[colored red], and this is **:blue[colored]** and bold."")

if  (strz=="nan") or (stre=="nan") or (strm=="nan"):
    st.markdown("  Aucun résults n'est disponible car : ")
    if  (strz=="nan"):
        st.markdown(" * Aucune :red[ZONE géographique] n'est choisie")
    if  (stre=="nan"):
        st.markdown(" * Aucune :red[METHODE de rééchantillonnage] n'est choisie")
    if  (strm=="nan"):
        st.markdown(" * Aucun :red[MODELE] n'est choisi")

else:
    # les scores du modèle :
    st.write(" * Score: ", 
        accuracy_score(y_test, y_pred_RF).round(2), 
        "--------------------" , 
        " Precision: ",
        precision_score(y_test, y_pred_RF).round(2),
        "--------------------" , 
        " Recall: ", 
            recall_score(y_test, y_pred_RF).round(2)
            )


        # matrice de confusion :
    st.write("* Matrice de Confusion")

    cf_matrix = confusion_matrix(y_test, y_pred_RF)
    group_names = ['T. Neg','F. Pos','F. Neg','T. Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    plt.rcParams["figure.figsize"] = [8, 5]
    res=sns.heatmap(cf_matrix, annot=labels, fmt='',linewidth=0.5, cmap='Greens',annot_kws={"size":16,'fontweight': 'bold'},cbar=False)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 16)
    res.set(xlabel='Classes prédites', ylabel='Classes réelles')
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 16)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    st.subheader("Comparaison Résultats")


    st.subheader("Choisir les modèles à comparer par zone geographique") # st.sidebar
    classifier_zz = st.selectbox(" ", ("--------------------","Modèles Zone A","Modèles Zone B","Modèles Zone C","Modèles Toutes les zones",
                                    "Tous les Modèles"))


    # Creating the DataFrame
    df = pd.DataFrame({'Zone':['A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride','A = Aride',
                               'B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée','B = Tempérée',
                               'C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale','C = Tropicale',
                               'Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones','Toutes les zones'],
                       'Algo. ré-équilibrage':[' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',
                               ' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',
                               ' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',
                               ' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling',' Sans','Oversampling','undersampling'],
                        'Modele':['KNN','KNN','KNN','SVM','SVM','SVM','RF','RF','RF','RN','RN','RN',
                               'KNN','KNN','KNN','SVM','SVM','SVM','RF','RF','RF','RN','RN','RN',
                               'KNN','KNN','KNN','SVM','SVM','SVM','RF','RF','RF','RN','RN','RN',
                               'KNN','KNN','KNN','SVM','SVM','SVM','RF','RF','RF','RN','RN','RN'],
                        'Score':[0.85,0.81,0.82,0.85,0.84,0.82,0.86,0.85,0.81,0.86,0.86,0.85,
                               0.91,0.88,0.79,0.92,0.85,0.84,0.91,0.92,0.82,0.92,0.91,0.86,
                               0.84,0.79,0.78,0.85,0.81,0.80,0.85,0.85,0.80,0.85,0.85,0.84,
                               0.84,0.80,0.78,0.86,0.81,0.80,0.86,0.86,0.81,0.85,0.85,0.85],
                        'Prediction':[0.8,0.64,0.61,0.77,0.64,0.59,0.78,0.73,0.57,0.77,0.74,0.69,
                               0.74,0.45,0.32,0.79,0.4,0.39,0.75,0.72,0.36,0.74,0.56,0.42,
                               0.74,0.56,0.52,0.78,0.57,0.55,0.77,0.72,0.55,0.76,0.69,0.68,
                               0.73,0.54,0.49,0.76,0.54,0.53,0.76,0.70,0.54,0.72,0.66,0.69],
                        'Recall':[0.5,0.57,0.78,0.58,0.83,0.86,0.58,0.63,0.87,0.62,0.65,0.69,
                               0.27,0.41,0.8,0.38,0.78,0.82,0.34,0.43,0.85,0.40,0.69,0.68,
                               0.46,0.50,0.77,0.49,0.79,0.80,0.51,0.60,0.81,0.50,0.62,0.61,
                               0.46,0.50,0.77,0.49,0.79,0.79,0.51,0.58,0.79,0.54,0.64,0.57]
                })

        # Create the index
    index_ = ['M_1', 'M_2', 'M_3', 'M_4', 'M_5','M_6', 'M_7', 'M_8', 'M_9', 'M_10',
            'M_11', 'M_12', 'M_13', 'M_14', 'M_15','M_16', 'M_17', 'M_18', 'M_19', 'M_20',
            'M_21', 'M_22', 'M_23', 'M_24', 'M_25','M_26', 'M_27', 'M_28', 'M_29', 'M_30',
            'M_31', 'M_32', 'M_33', 'M_34', 'M_35','M_36', 'M_37', 'M_38', 'M_39', 'M_40',
            'M_41', 'M_42', 'M_43', 'M_44', 'M_45','M_46', 'M_47', 'M_48']

        # Set the index
    df.index = index_

    dfA=df.iloc[0:12,:]
    dfB=df.iloc[12:24,:]
    dfC=df.iloc[24:36,:]
    dfAll=df.iloc[36:48,:]

    if classifier_zz == "--------------------":

        st.write("")

    if classifier_zz == "Tous les Modèles":

        st.table(df.style.highlight_min(subset=['Score',"Prediction","Recall"],color='lightcoral',axis=0).highlight_max(subset=['Score',"Prediction","Recall"],color='forestgreen',axis=0).format('{:.2f}',subset=['Score',"Prediction","Recall"]))

    if classifier_zz == "Modèles Zone A":

        st.table(dfA.style.highlight_min(subset=['Score',"Prediction","Recall"],color='lightcoral',axis=0).highlight_max(subset=['Score',"Prediction","Recall"],color='forestgreen',axis=0).format('{:.2f}',subset=['Score',"Prediction","Recall"]))

    if classifier_zz == "Modèles Zone B":

        st.table(dfB.style.highlight_min(subset=['Score',"Prediction","Recall"],color='lightcoral',axis=0).highlight_max(subset=['Score',"Prediction","Recall"],color='forestgreen',axis=0).format('{:.2f}',subset=['Score',"Prediction","Recall"]))

    if classifier_zz == "Modèles Zone C":

        st.table(dfC.style.highlight_min(subset=['Score',"Prediction","Recall"],color='lightcoral',axis=0).highlight_max(subset=['Score',"Prediction","Recall"],color='forestgreen',axis=0).format('{:.2f}',subset=['Score',"Prediction","Recall"]))

    if classifier_zz == "Modèles Toutes les zones":

        st.table(dfAll.style.highlight_min(subset=['Score',"Prediction","Recall"],color='lightcoral',axis=0).highlight_max(subset=['Score',"Prediction","Recall"],color='forestgreen',axis=0).format('{:.2f}',subset=['Score',"Prediction","Recall"]))

    

st.markdown(
    """



    """
)
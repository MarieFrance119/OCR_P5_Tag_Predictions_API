#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:07:27 2022

@author: MFLB
"""

# Déploiement sur Streamlite
import streamlit as st

# Parsing des données texte
from bs4 import BeautifulSoup

# Librairie nltk pour traiter les mots
import nltk
# for API deployment
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Outils de sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

# Data Persistence
import joblib

# Fonctions utiles pour prédiction

# Pré-traitement du texte
def get_preprocessed_text(raw_question,raw_text):
    
    """
    Fonction pour traiter une question et un post
    en une liste de mots utilisable pour la modélisation
    
    - Arguments :
        - raw_question : question d'origine
        - raw_text : texte d'origine
    
    - Retourne :
        - text : le texte pré-traité, prêt pour la modélisation
    """
    
    # fusion de la question et du post
    text = raw_question + " " + raw_text
    
    # Parsage du texte et supprimer balises html
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Tokenizer pour récupérer que les termes avec des lettres
    tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]+")

    # Récupération de la liste des mots mis en minuscule
    tokens = tokenizer.tokenize(text.lower())

    # On ne garde que les mots d'au moins 3 lettres
    tokens = list(filter(lambda x: len(x) >= 3, tokens))

    # Récupération des stopwords English
    sw = set(stopwords.words("english"))

    # Supprimer les stop words
    token_cleaned = [token for token in tokens if not token in sw]
    
    # POS tagging
    pos_t = nltk.pos_tag(token_cleaned)

    # Récupération des noms communs
    token_noun = [token[0] for token in pos_t if token[1] in ("NN", "NNS")]
    
    # Initialisation du lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Récupération des lems
    lemmed_text = [lemmatizer.lemmatize(token) for token in token_noun]
    
    return lemmed_text


# Prédiction des tags
def predict_tags(lemmed_text):

    """ Fonction pour afficher la liste des tags prédits par le
    modèle
    
    - Arguments :
        - lemmed_text : texte pré-processé dont il faut prédire les tags
    
    - Retourne:
         - pred_tags_list : la liste des tags, en filtrant sur les mots 
         présents dans le texte
    """

    # Import du modèle
    pipeline_SVM_10 = joblib.load("./models/pipeline_SVM_10.joblib")
   
    # Application du modèle
    pred_tags = pipeline_SVM_10.predict(lemmed_text)
    
    return pred_tags

# transformation des tags vectorisés en "mots"
def get_tags(pred_tags, lemmed_text):

    """ Fonction pour afficher la liste des tags prédits par le
    modèle
    
    - Arguments :
        - pred_tags : tags prédits en sortie du modèle
        - lemmed_text : texte pré-processé dont il faut prédire les tags
    
    - Retourne:
         - pred_tags_list : la liste des tags, en filtrant sur les mots 
         présents dans le texte
    """

    # import du MultilabelBinarizer
    mlb = joblib.load("./models/mlb_model.joblib")

    # Récupération des labels
    text_tags = mlb.inverse_transform(pred_tags)

    # Liste des tags
    pred_tags_list = list(
        {tag for tag_list in text_tags for tag in tag_list if (len(tag_list) != 0)}
    )

    # Filtrage sur les mots présents dans le lemmed_text
    pred_tags_list = [tag for tag in pred_tags_list if tag in lemmed_text]

    return ", ".join(pred_tags_list)

# Fonction pour streamlite
def main():

    # Affichage du titre
    st.title('Tags Prediction for StackOverflow Questions')

    # Input à saisir
    question_text = st.text_input('Enter your question ',
                              value="write here")
    description_text = st.text_area('Give more information ', height = 300,
                              value="write here")

    # Récupération du texte traité à envoyer dans l'API et du texte sous
    # forme de lems
    preprocessed_text = get_preprocessed_text(question_text, description_text)

    # Bouton pour prédiction
    predict_btn = st.button('Prediction')
    
    if predict_btn:

        # Prédiction des tages
        pred = predict_tags(preprocessed_text)
        
        # Transformation en liste de tags
        list_tag = get_tags(pred, preprocessed_text)

        # Affichage de la réponse
        st.write(
            'Predicted tags are : {}'.format(list_tag))


if __name__ == '__main__':
    main()

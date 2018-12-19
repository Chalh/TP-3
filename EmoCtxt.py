# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-3
# DATE:                                                                                                 #
########################################################################################################################

import pandas as pd
import sklearn
import numpy as np
import re
import os
from sklearn.metrics import recall_score
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from nltk.corpus import wordnet as wn

from nltk import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
import sys

ps = PorterStemmer()

stopwd  = set(sw.words('english'))

lemmatizer = WordNetLemmatizer()

#fonction pour:
#             - faire la lemmatization, le stemming d'un mot ou RIEN FAIRE selon la variable "normalize"


def Normaliser_mot( token, tag, normalize):

    if tag is None:
        return None
    else:
        try:
            if normalize == 0: #RIEN FAIRE
                return token
            else:
                if normalize == 1: #STEMMING
                    return ps.stem(token)
                else:           # Lemmatization
                    tag = {
                        'N': wn.NOUN,
                        'V': wn.VERB,
                        'R': wn.ADV,
                        'J': wn.ADJ
                    }.get(tag[0])

                    return lemmatizer.lemmatize(token, tag)

        except:
            return None

#Fonction pour préparer la liste d'attribut d'une phrase:
#   -Enlever les stop words
#   - Enlever les signes de ponctuations
def Tranforme_texte(texte_st, normalize, ponctuation=False):
    resultat=[]
    # Convertir en miniscule
    try:
        document_tok = sent_tokenize(str(texte_st).lower())
    except:
        document_tok = sent_tokenize(str("this is a neutral sentence"))

    doc_res = []
    for sent in document_tok:
        #Enlever la ponctuation mais garder les emoticons (2 signes de ponc qui se suivent)
        if ponctuation is True:
            #On enlève seulement les signes de
            sent = re.sub(r'((?<=\w)[^\s\w](?![^\s\w]))', '', sent)
        set_res = []
        for token, tag in pos_tag(word_tokenize(sent)):
            #Appliquer la normalisation
            resultat_norm = Normaliser_mot(token, tag,normalize)
            if resultat_norm is None:
                resultat_norm = token
            set_res.append(resultat_norm)
        set_res = ' '.join(set_res)
        doc_res.append(set_res)
    doc_res = ' '.join(doc_res)
    return doc_res


#Fonction pour préparer le fihchier CSV "propre" de données d'apprentissage. Chaque phrase (turn) est une entrée
# dans le fichier d'apresentissage
#Les lignes sont constituées ainsi:
#
#Index  |Text               |Target
#  1    |turn1 de ligne 1   |Emotion de la ligne1
#  2    |turn2 de ligne 1   |Emotion de la ligne1
#  3    |turn3 de ligne 1   |Emotion de la ligne1
#  4    |turn1 de ligne 2   |Emotion de la ligne2
#  ...
#  1    |turn3 de ligne N   |Emotion de la ligneN

def Generer_CSV_Entrain_propre_UnTweetAlafois(fichierin ="corpus/train.txt", fichierout="corpus/train_clean_1T.txt",separt = "\t", normalise = 1,rem_pct=False ):
    try:
        inf_train = open(fichierin, "r")
        outf_train = open(fichierout, "w")
        xdata_train = pd.read_csv(inf_train, sep=separt)
        xyz = xdata_train.values
        outf_train.write("ndex\ttext\ttarget\n")
        nb_id = 0
        for i in xyz:
            for j in range(1,4):
                ligne = str(nb_id)+"\t"+Tranforme_texte(str(i[j]),normalise,rem_pct)+"\t"+str(i[4])+"\n"
                nb_id = nb_id  + 1
                outf_train.write(ligne.__str__())
    except:
        return False
    return True

#Fonction pour préparer le fihchier CSV "propre" de données d'apprentissage. Chaque phrase est la concaténation
# des trois phrase (turn1<espace>turn2<espace>turn3)
# dans le fichier d'apresentissage
#Les lignes sont constituées ainsi:
#
#Index  |Text               |Target
#  1    |turn1 turn2 turn3 de ligne 1   |Emotion de la ligne1
#  1    |turn1 turn2 turn3 de ligne 2   |Emotion de la ligne2
#  ...
#  1    |turn1 turn2 turn3 de ligne N   |Emotion de la ligneN
def Generer_CSV_Entrain_propre_TroisTweetAlafois(fichierin ="corpus/train.txt", fichierout="corpus/train_clean_3T.txt",separt = "\t" , normalise = 1 ,rem_pct=False ):
    try:
        inf_train = open(fichierin, "r")
        outf_train = open(fichierout, "w")
        xdata_train = pd.read_csv(inf_train, sep=separt)
        xyz = xdata_train.values
        outf_train.write("index\ttext\ttarget\n")
        nb_id = 0
        for i in xyz:
            ligne = str(nb_id)+"\t"+Tranforme_texte(" ".join(i[1:4]),normalise,rem_pct)+"\t"+str(i[4])+"\n"
            nb_id = nb_id  + 1
            outf_train.write(ligne.__str__())
    except:
        return False
    return True

def Calculer_Precision(pipeline, x_train, y_train, x_test, y_test):

    t0 = time()
    emotion_fit = pipeline.fit(x_train, y_train)
    y_pred = emotion_fit.predict(x_test)
    train_test_time = "{0:.2f}".format(time() - t0)
    precision = "{0:.2f}".format((accuracy_score(y_test, y_pred))*100)
    print(precision)
    return precision, train_test_time


def Tester_Modele_NbAttributs(x,y,vectorizer=CountVectorizer(), attributs=np.arange(3000,30001,3000), stop_words=None, ngram_range=1, classifier=LogisticRegression()):
    print("-" * 100)
    print( (vectorizer))
    print( "\n")
    print(ngram_range)
    print("\n")
    print("-" * 100)
    result = []
    SEED = 2000
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.02,random_state=SEED)
    ngrame_rg = (1,ngram_range)
    for n in attributs:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngrame_rg)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print( "Validation des résultats pour {} attributs".format(n))
        nfeature_accuracy,tt_time = Calculer_Precision(checker_pipeline, x_train, y_train, x_test, y_test)
        result.append((str(ngram_range),str(n),nfeature_accuracy,tt_time))

    print("-" * 100)
    return result


def Générer_fichier_resultat(nom_csv_atester, seprt="\t",nom_repertoirecible="corpus",stop_wd="AvecSW", nom_algo="LR", nom_vectzer="C-VECT"):

    resultats =[]
    print(nom_csv_atester+"::"+nom_algo+"::"+nom_vectzer)
    fichier_train = nom_repertoirecible + "/" + nom_csv_atester
    my_df = pd.read_csv(fichier_train, index_col=0, sep=seprt)

    if "_AP" in nom_csv_atester:
        nom_repertoirecible += "/AP"
    else:
        if "_SP" in nom_csv_atester:
            nom_repertoirecible += "/SP"
        else:
            return False
    if not os.path.exists(nom_repertoirecible):
        os.makedirs(nom_repertoirecible)


    if "_1T_" in nom_csv_atester:
        nom_repertoirecible += "/1T"
    else:
        if "_3T_" in nom_csv_atester:
            nom_repertoirecible += "/3T"
        else:
            return False

    if not os.path.exists(nom_repertoirecible):
        os.makedirs(nom_repertoirecible)

    if nom_algo == "LR":
        algo = LogisticRegression()
        nom_repertoirecible += "/LR"
    else:
        algo = MultinomialNB()
        nom_repertoirecible += "/MB"

    if not os.path.exists(nom_repertoirecible):
        os.makedirs(nom_repertoirecible)


    if nom_vectzer == "C-VECT":
        vectzer = CountVectorizer()
    else: #TFIDF
        vectzer = TfidfVectorizer()

    st_wd = None

    if stop_wd == "AvecSW":
        st_wd = None
    else: #"EnleverSW"
        st_wd = "english"

    nom_fich_res = nom_repertoirecible+ "/"+nom_csv_atester + stop_wd + nom_algo + nom_vectzer + ".csv"

    resultats = Tester_Modele_NbAttributs(x=my_df.text, y=my_df.target, vectorizer=vectzer, attributs=n_attributs, stop_words=st_wd, classifier=algo) + \
                Tester_Modele_NbAttributs(x=my_df.text, y=my_df.target, vectorizer=vectzer, attributs=n_attributs, stop_words=st_wd, classifier=algo, ngram_range=2) + \
                Tester_Modele_NbAttributs(x=my_df.text, y=my_df.target, vectorizer=vectzer, attributs=n_attributs, stop_words=st_wd, classifier=algo, ngram_range=3)

    exists = os.path.isfile(nom_fich_res)
    if exists is True:
        os.remove(nom_fich_res)

    file = open(nom_fich_res, "w")

    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    return True

def Creer_fichier_train(csv_entree, file, nrm = 0,rmpct=False):

    exists = os.path.isfile(file)
    if exists is True:
        print("Le fichier \""+file+" \"existe déja!")
    else:
        print("Création du fichier \"" + file + " \"")
        if "_1T_" in file:
            cleandata = Generer_CSV_Entrain_propre_UnTweetAlafois(csv_entree, file, normalise=nrm, rem_pct=rmpct)
        else:
            cleandata = Generer_CSV_Entrain_propre_TroisTweetAlafois(csv_entree, file, normalise=nrm,rem_pct=rmpct)

        if cleandata is False:
            print("Erreur de création du fichier \""+file+" \" !")



csvin ="corpus/train.txt"

#noms des fichies entrainement
csvout_untweet_AvecPonct = ["train_cln_1T_nrm_0_AP.txt", "train_cln_1T_nrm_1_AP.txt", "train_cln_1T_nrm_2_AP.txt"]
csvout_Troistweet_AvecPonct = ["train_cln_3T_nrm_0_AP.txt", "train_cln_3T_nrm_1_AP.txt", "train_cln_3T_nrm_2_AP.txt"]

csvout_untweet_SansPonct = ["train_cln_1T_nrm_0_SP.txt", "train_cln_1T_nrm_1_SP.txt", "train_cln_1T_nrm_2_SP.txt"]
csvout_Troistweet_SansPonct = ["train_cln_3T_nrm_0_SP.txt", "train_cln_3T_nrm_1_SP.txt", "train_cln_3T_nrm_2_SP.txt"]

repertoire_corpus = "corpus"

# ########################################################################################################################
#                                                                                                                      #
#                           Générer les fichiers d'entrainement
# #
########################################################################################################################


for xtr in range(0,3):
    Creer_fichier_train(csvin, repertoire_corpus + "/"+ csvout_untweet_AvecPonct[xtr],nrm = xtr)
    Creer_fichier_train(csvin, repertoire_corpus + "/" + csvout_Troistweet_AvecPonct[xtr], nrm=xtr)
    Creer_fichier_train(csvin, repertoire_corpus + "/" + csvout_untweet_SansPonct[xtr], nrm=xtr,rmpct=True)
    Creer_fichier_train(csvin, repertoire_corpus + "/" + csvout_Troistweet_SansPonct[xtr], nrm=xtr,rmpct=True)

# ########################################################################################################################
#                                                                                                                      #
#                           Tester les différentes situations
# #
########################################################################################################################

n_attributs = np.arange(3000,30001,3000)
#n_attributs = np.arange(100,1001,100)

Algo_Name = ["LR","MB"]

Vectorizer_Name = ["C-VECT","TF-IDF"]

SW_test_Name = ["AvecSW","EnleverSW"]




#################################
#   Tester avec ponctuation
#################################

print("Tester avec ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")

for i in range(0,3):
    for m_alg in Algo_Name:
        for m_vec in Vectorizer_Name:
            for stp_wdr in SW_test_Name:
                Générer_fichier_resultat(csvout_untweet_AvecPonct[i], seprt="\t", nom_repertoirecible=repertoire_corpus, stop_wd=stp_wdr,
                                         nom_algo=m_alg, nom_vectzer=m_vec)

                Générer_fichier_resultat(csvout_Troistweet_AvecPonct[i], seprt="\t", nom_repertoirecible=repertoire_corpus, stop_wd=stp_wdr,
                                         nom_algo=m_alg, nom_vectzer=m_vec)

print("Tester sans ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")

for i in range(0, 3):
    for m_alg in Algo_Name:
        for m_vec in Vectorizer_Name:
            for stp_wdr in SW_test_Name:
                Générer_fichier_resultat(csvout_untweet_SansPonct[i], seprt="\t", nom_repertoirecible=repertoire_corpus,
                                         stop_wd=stp_wdr,
                                         nom_algo=m_alg, nom_vectzer=m_vec)

                Générer_fichier_resultat(csvout_Troistweet_SansPonct[i], seprt="\t", nom_repertoirecible=repertoire_corpus,
                                         stop_wd=stp_wdr,
                                         nom_algo=m_alg, nom_vectzer=m_vec)


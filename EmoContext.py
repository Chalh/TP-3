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
            sent = re.sub(r'((?<=\w)[^\s\w](?![^\s\w]))|(\.\.*)|(\?\?*)|(\!\!*)', '', sent)

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


def Tester_Modele_NbAttributs(vectorizer=CountVectorizer(), attributs=np.arange(3000,30001,3000), stop_words=None, ngram_range=1, classifier=LogisticRegression()):
    print("-" * 100)
    print( (vectorizer))
    print( "\n")
    print(ngram_range)
    print("\n")
    print("-" * 100)
    result = []
    ngrame_rg = (1,ngram_range)
    for n in attributs:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngrame_rg)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print( "Validation des résultats pour {} attributs".format(n))
        nfeature_accuracy,tt_time = Calculer_Precision(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((str(ngram_range),str(n),nfeature_accuracy,tt_time))

    print("-" * 100)
    return result


csvin ="corpus/train.txt"

#noms des fichies entrainement
csvout_untweet_AvecPonct = ["corpus/train_cln_1T_nrm_1_AP.txt", "corpus/train_cln_1T_nrm_2_AP.txt", "corpus/train_cln_1T_nrm_3_AP.txt"]
csvout_Troistweet_AvecPonct = ["corpus/train_cln_3T_nrm_1_AP.txt", "corpus/train_cln_3T_nrm_2_AP.txt", "corpus/train_cln_3T_nrm_3_AP.txt"]

csvout_untweet_SansPonct = ["corpus/train_cln_1T_nrm_1_SP.txt", "corpus/train_cln_1T_nrm_2_SP.txt", "corpus/train_cln_1T_nrm_3_SP.txt"]
csvout_Troistweet_SansPonct = ["corpus/train_cln_3T_nrm_1_SP.txt", "corpus/train_cln_3T_nrm_2_SP.txt", "corpus/train_cln_3T_nrm_3_SP.txt"]


#noms des fichies resultats
csvout_untweet_AvecPonct_rslt = ["corpus/train_cln_1T_nrm_1_AP_resultat", "corpus/train_cln_1T_nrm_2_AP_resultat", "corpus/train_cln_1T_nrm_3_AP_resultat"]
csvout_Troistweet_AvecPonct_rslt = ["corpus/train_cln_3T_nrm_1_AP_resultat", "corpus/train_cln_3T_nrm_2_AP_resultat", "corpus/train_cln_3T_nrm_3_AP_resultat"]

csvout_untweet_SansPonct_rslt = ["corpus/train_cln_1T_nrm_1_SP_resultat", "corpus/train_cln_1T_nrm_2_SP_resultat", "corpus/train_cln_1T_nrm_3_SP_resultat"]
csvout_Troistweet_SansPonct_rslt = ["corpus/train_cln_3T_nrm_1_SP_resultat", "corpus/train_cln_3T_nrm_2_SP_resultat", "corpus/train_cln_3T_nrm_3_SP_resultat"]


# ########################################################################################################################
#                                                                                                                      #
#                           Générer les fichiers d'entrainement
# #
########################################################################################################################




for x in range(0,3):
    exists = os.path.isfile(csvout_untweet_AvecPonct[x])
    if exists is True:
        print("Le fichier \""+csvout_untweet_AvecPonct[x]+" \"existe déja!")
    else:
        print("Création du fichier \"" + csvout_untweet_AvecPonct[x] + " \"")
        cleandata = Generer_CSV_Entrain_propre_UnTweetAlafois(csvin, csvout_untweet_AvecPonct[x], normalise=x)
        if cleandata is False:
            print("Erreur de création du fichier \""+csvout_untweet_AvecPonct[x]+" \" !")

    exists = os.path.isfile(csvout_untweet_SansPonct[x])
    if exists is True:
        print("Le fichier \""+csvout_untweet_SansPonct[x]+" \"existe déja!")
    else:
        print("Création du fichier \"" + csvout_untweet_SansPonct[x] + " \"")
        cleandata = Generer_CSV_Entrain_propre_UnTweetAlafois(csvin, csvout_untweet_SansPonct[x], normalise=x,rem_pct=True)
        if cleandata is False:
            print("Erreur de création du fichier \""+csvout_untweet_SansPonct[x]+" \" !")

    exists = os.path.isfile(csvout_Troistweet_AvecPonct[x])
    if exists is True:
        print("Le fichier \"" + csvout_Troistweet_AvecPonct[x] + " \"existe déja!")
    else:
        print("Création du fichier \"" + csvout_Troistweet_AvecPonct[x] + " \"")
        cleandata = Generer_CSV_Entrain_propre_TroisTweetAlafois(csvin, csvout_Troistweet_AvecPonct[x], normalise=x)
        if cleandata is False:
            print("Erreur de création du fichier \""+csvout_Troistweet_AvecPonct[x]+" \" !")

    exists = os.path.isfile(csvout_Troistweet_SansPonct[x])
    if exists is True:
        print("Le fichier \"" + csvout_Troistweet_SansPonct[x] + " \"existe déja!")
    else:
        print("Création du fichier \"" + csvout_Troistweet_SansPonct[x] + " \"")
        cleandata = Generer_CSV_Entrain_propre_TroisTweetAlafois(csvin, csvout_Troistweet_SansPonct[x], normalise=x,rem_pct=True)
        if cleandata is False:
            print("Erreur de création du fichier \""+csvout_Troistweet_SansPonct[x]+" \" !")


# ########################################################################################################################
#                                                                                                                      #
#                           Tester les différentes situations
# #
########################################################################################################################
lr = LogisticRegression()
print("Logistic regression")
n_attributs = np.arange(100,1001,100)


#############################################################################################################
CVECTtx = "CountVect_LR"


#################################
#   Tester avec ponctuation
#################################

print("Tester avec ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
cvec = CountVectorizer()

#(1T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_untweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_untweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_untweet_AvecPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_untweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#(3T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_Troistweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_Troistweet_AvecPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_Troistweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#################################
#   Tester sans ponctuation
#################################
print("Tester sans ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
#(1T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_untweet_SansPonct[i])
    my_df = pd.read_csv(csvout_untweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_untweet_SansPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_untweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#(3T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_Troistweet_SansPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_Troistweet_SansPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_Troistweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

###############################################################################################

CVECTtx = "TFIDF_LR"

print()
#################################
#   Tester avec ponctuation
#################################

print("Tester avec ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
cvec = CountVectorizer()



#(1T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_untweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_untweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_untweet_AvecPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_untweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#(3T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_Troistweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_Troistweet_AvecPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_Troistweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#################################
#   Tester sans ponctuation
#################################
print("Tester sans ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
#(1T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_untweet_SansPonct[i])
    my_df = pd.read_csv(csvout_untweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_untweet_SansPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_untweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#(3T (NORM:1 a 3)
for i in range(0,3):
    print(csvout_Troistweet_SansPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,attributs=n_attributs,ngram_range=3)

    fich_res = csvout_Troistweet_SansPonct_rslt[i]+"garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs) + \
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=2)+\
                                   Tester_Modele_NbAttributs(vectorizer=cvec,stop_words="english",attributs=n_attributs,ngram_range=3)


    fich_res = csvout_Troistweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res,"w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0]+ "\t"+xa[1]+ "\t"+xa[2]+ "\t"+xa[3]+ "\n"
        file.write(sent)

#############################################################################################################
#############################################################################################################
#############################################################################################################

lr = MultinomialNB()
print("MultinomialNB")

#############################################################################################################
CVECTtx = "CountVect_MB"

#################################
#   Tester avec ponctuation
#################################

print("Tester avec ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
cvec = CountVectorizer()

# (1T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_untweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_untweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_untweet_AvecPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_untweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

# (3T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_Troistweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_Troistweet_AvecPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_Troistweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

#################################
#   Tester sans ponctuation
#################################
print("Tester sans ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
# (1T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_untweet_SansPonct[i])
    my_df = pd.read_csv(csvout_untweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_untweet_SansPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_untweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

# (3T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_Troistweet_SansPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_Troistweet_SansPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_Troistweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

###############################################################################################

CVECTtx = "TFIDF_MB"

print()
#################################
#   Tester avec ponctuation
#################################

print("Tester avec ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
cvec = CountVectorizer()

# (1T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_untweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_untweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_untweet_AvecPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_untweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

# (3T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_Troistweet_AvecPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_AvecPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_Troistweet_AvecPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_Troistweet_AvecPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

#################################
#   Tester sans ponctuation
#################################
print("Tester sans ponctuation (1T (NORM:1 a 3) et 3T (NORM:1 a 3))")
# (1T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_untweet_SansPonct[i])
    my_df = pd.read_csv(csvout_untweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_untweet_SansPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_untweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

# (3T (NORM:1 a 3)
for i in range(0, 3):
    print(csvout_Troistweet_SansPonct[i])
    my_df = pd.read_csv(csvout_Troistweet_SansPonct[i], index_col=0, sep="\t")
    x = my_df.text[:1000]
    y = my_df.target[:1000]

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02,
                                                                                      random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                  test_size=.5, random_state=SEED)

    resultats_garderstropword = Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=2) + \
                                Tester_Modele_NbAttributs(vectorizer=cvec, attributs=n_attributs, ngram_range=3)

    fich_res = csvout_Troistweet_SansPonct_rslt[i] + "garderSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

    resultats_enleverstropword = Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english",
                                                           attributs=n_attributs) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=2) + \
                                 Tester_Modele_NbAttributs(vectorizer=cvec, stop_words="english", attributs=n_attributs,
                                                           ngram_range=3)

    fich_res = csvout_Troistweet_SansPonct_rslt[i] + "enleverSW_" + CVECTtx
    exists = os.path.isfile(fich_res)
    if exists is True:
        os.remove(fich_res)

    file = open(fich_res, "w")
    file.write("ngramme\tnfeatures\tprecision\ttest_time\n")
    for xa in resultats_garderstropword:
        sent = xa[0] + "\t" + xa[1] + "\t" + xa[2] + "\t" + xa[3] + "\n"
        file.write(sent)

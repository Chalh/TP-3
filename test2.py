# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-2
# DATE:                                                                                                 #
########################################################################################################################

import pandas as pd
import sklearn
import numpy as np

from sklearn.metrics import recall_score
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

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

def lemmatize_texte( token, tag, normalize):

    if tag is None:
        return None
    else:

        try:
            if normalize == 1:
                tag = {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0])

                return lemmatizer.lemmatize(token, tag)

            else:
                if normalize == 2:
                    return ps.stem(token)
                else:
                    return token
        except:
            return None


def Tranforme_texte(texte_st, normalize):


    resultat=[]

    # Converting to Lowercase
    try:
        document_tok = sent_tokenize(str(texte_st).lower())
    except:
        texte_st = "good bad "
        document_tok = sent_tokenize(str(texte_st).lower())
    doc_res = []
    for sent in document_tok:
        # Break the sentence into part of speech tagged tokens
        set_res = []
        for token, tag in pos_tag(word_tokenize(sent)):
            # Apply preprocessing to the token
            token = token.lower()

            # Si stopword, ignorer le token et continuer
            if token in stopwd:
                continue

            # Lem#matize the token and yield
            lemma = lemmatize_texte(token, tag,normalize)
            if lemma is None:
                continue
            set_res.append(lemma)

        set_res = ' '.join(set_res)

        doc_res.append(set_res)
    doc_res = ' '.join(doc_res)
    return doc_res

def shuffle(matrix, target, test_proportion):
    ratio = (matrix.shape[0]/test_proportion).__int__()
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:]
    Y_test =  target[:ratio]
    return X_train, X_test, Y_train, Y_test


f_train = open("corpus/train.txt","r")
f_test = open("corpus/devwithoutlabels.txt","r")

data_train = pd.read_csv(f_train,sep="\t")
data_test = pd.read_csv(f_test,sep="\t")


data_train = data_train.reindex(np.random.permutation(data_train.index))
data_test = data_test.reindex(np.random.permutation(data_test.index))

Nb_col = 1000

def Generer_CSV_Entrain_propre(fichierin ="corpus/train.txt", fichierout="corpus/train_clean.txt",separt = "\t" ):
    try:
        inf_train = open(fichierin, "r")
        outf_train = open(fichierout, "w")
        xdata_train = pd.read_csv(inf_train, sep=separt)
        xyz = xdata_train.values
        outf_train.write("index\ttext\ttarget\n")
        nb_id = 0
        for i in xyz:
            for j in range(1,4):
                ligne = str(nb_id)+"\t"+str(i[j])+"\t"+str(i[4])+"\n"
                nb_id = nb_id  + 1
                outf_train.write(ligne.__str__())
    except:
        return False
    return True

def Generer_CSV_Entrain_propre2(fichierin ="corpus/train.txt", fichierout="corpus/train_clean.txt",separt = "\t" ):
    try:
        inf_train = open(fichierin, "r")
        outf_train = open(fichierout, "w")
        xdata_train = pd.read_csv(inf_train, sep=separt)
        xyz = xdata_train.values
        outf_train.write("index\ttext\ttarget\n")
        nb_id = 0
        for i in xyz:
            ligne = str(nb_id)+"\t"+Tranforme_texte(" ".join(i[1:4]),2)+"\t"+str(i[4])+"\n"
            nb_id = nb_id  + 1
            outf_train.write(ligne.__str__())
    except:
        return False
    return True

csvin ="corpus/train.txt"
csvout = "corpus/train_clean4.txt"

#cleandata = Generer_CSV_Entrain_propre(csvin,csvout)
cleandata = Generer_CSV_Entrain_propre2(csvin,csvout)
#if cleandata is True:
#    print("OK")
#else:
#    print("NOKhhhjhjhcdhcjdh")

my_df = pd.read_csv(csvout,index_col=0,sep="\t")
print(my_df.head())

my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()

x = my_df.text
y = my_df.target



xdata_train = xdata_train.reindex(np.random.permutation(xdata_train.index))

DX_train = []
DY_train = []
#xyz = data_train.values[:Nb_col]
xyz = data_train.values
for i in xyz:
    phrase = " ".join(i[1:4])
    DX_train.append(phrase)
    DY_train.append(i[4])

DX_test = []

#xyz = data_test.values[:Nb_col]
xyz = data_test.values
for i in xyz:
    phrase = " ".join(i[1:4])
    DX_test.append(phrase)


print("ok1")
mindf = 1
normalisation = 2
documents = []
for sen in DX_train:
    documents.append(Tranforme_texte(sen,normalisation))

documents_test = []
for sen in DX_test:
    documents_test.append(Tranforme_texte(sen,normalisation))

print("ok2")
print("ok2")
vectorizer = CountVectorizer(min_df=mindf, stop_words=stopwd)
#vectorizer = TfidfVectorizer(min_df=mindf, stop_words=stopwd)

X = vectorizer.fit_transform(documents)
XT = vectorizer.transform(documents_test)
x_train, x_test, y_train, y_test = shuffle(X, DY_train, 4)

clf = MultinomialNB().fit(x_train, y_train)
y_pred = clf.predict(x_test)
dy_pred = clf.predict(XT)

# 3. fit
#logreg.fit(X_train, y)

#average_precision_nb = average_precision_score(y_test, y_pred)
accuracy_score_nb = sklearn.metrics.accuracy_score(y_test, y_pred)
#recall_nb = recall_score(y_test, y_pred)
print(accuracy_score_nb)

logreg = LogisticRegression().fit(x_train, y_train)
lgy_pred = logreg.predict(x_test)
dlgy_pred = logreg.predict(XT)
#average_precision_lr = average_precision_score(y_test, y_pred)
#recall_lg = recall_score(y_test, y_pred)
accuracy_score_lr = sklearn.metrics.accuracy_score(y_test, lgy_pred)
print(accuracy_score_lr)

f = open('Outnormalisationstemaaa','w')
stdout_old = sys.stdout
sys.stdout = f
print("turn1-turn 2-turn 3 ;NAIVE BAYES;LOGISTIC REG")
i=0
#for x in data_test.values:
#    print(x[1].__str__()+";"+x[2].__str__()+";"+x[3].__str__()+";"+dy_pred[i] +";"+dlgy_pred[i])
#    i +=1


for x in x_test:
    print(x.__str__()+";"+y_pred[i] +";"+lgy_pred[i])
    i +=1
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

def shuffle_list(liste, target, test_proportion):
    ratio = (len(liste)/test_proportion).__int__()
    lx_train = liste[ratio:]
    lx_test =  liste[:ratio]
    ly_train = target[ratio:]
    ly_test =  target[:ratio]
    return lx_train, lx_test, ly_train, ly_test


f_train = open("corpus/train.txt","r")
f_test = open("corpus/devwithoutlabels.txt","r")

data_train = pd.read_csv(f_train,sep="\t")
data_test = pd.read_csv(f_test,sep="\t")


data_train = data_train.reindex(np.random.permutation(data_train.index))
data_test = data_test.reindex(np.random.permutation(data_test.index))

Nb_col = 1000


X_train = []
Y_train = []

xyz = data_train.values
for i in xyz:
    phrase = " ".join(i[1:4])
    X_train.append(phrase)
    Y_train.append(i[4])

DX_train, DX_test, DY_train, DY_test = shuffle_list(X_train, Y_train, 4)


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

vectorizer = CountVectorizer(min_df=mindf, stop_words=stopwd)
#vectorizer = TfidfVectorizer(min_df=mindf, stop_words=stopwd)

X = vectorizer.fit_transform(documents)
XT = vectorizer.transform(documents_test)

clf = MultinomialNB().fit(X, DY_train)
dy_pred = clf.predict(XT)

# 3. fit
#logreg.fit(X_train, y)

#average_precision_nb = average_precision_score(y_test, y_pred)
accuracy_score_nb = sklearn.metrics.accuracy_score(DY_test, dy_pred)
#recall_nb = recall_score(y_test, y_pred)
print(accuracy_score_nb)

logreg = LogisticRegression().fit(X, DY_train)
dlgy_pred = logreg.predict(XT)
#average_precision_lr = average_precision_score(y_test, y_pred)
#recall_lg = recall_score(y_test, y_pred)
accuracy_score_lr = sklearn.metrics.accuracy_score(DY_test, dlgy_pred)
print(accuracy_score_lr)

f = open('abc','w')
stdout_old = sys.stdout
sys.stdout = f
print("turn1-turn 2-turn 3 ;NAIVE BAYES;LOGISTIC REG")
i=0
#for x in data_test.values:
#    print(x[1].__str__()+";"+x[2].__str__()+";"+x[3].__str__()+";"+dy_pred[i] +";"+dlgy_pred[i])
#    i +=1


for x in DX_test:
    print(x.__str__()+";"+DY_test[i]+";"+dy_pred[i] +";"+dlgy_pred[i])
    i +=1
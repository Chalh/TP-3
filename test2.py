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

from sklearn.metrics import recall_score
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from nltk.corpus import wordnet as wn

from nltk import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer

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








f = open("corpus/train.txt","r")

data_train = pd.read_csv(f,sep="\t")

X_train = []

#print (data_train)

feature_cols = data_train.label

y = data_train.values

X_train = []
Y_train = feature_cols.values[:10]

for i in y:
    turn1 = i[1]
    turn2 = i[2]
    turn3 = i[3]
    phrase = " ".join(i[1:4])
    X_train.append(phrase)

print("ok1")
mindf = 2
documents = []
for sen in range(0, 10):
#for sen in range(0, len(X_train)):
    documents.append(Tranforme_texte(X_train[sen],3))

from sklearn.linear_model import LogisticRegression

# 2. instantiate model
logreg = LogisticRegression()
print("ok2")
vectorizer = CountVectorizer(min_df=mindf, stop_words=stopwd)
X = vectorizer.fit_transform(documents).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
clf = MultinomialNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 3. fit
#logreg.fit(X_train, y)

average_precision_nb = average_precision_score(y_test, y_pred)
accuracy_score_nb = sklearn.metrics.accuracy_score(y_test, y_pred)
recall_nb = recall_score(y_test, y_pred)

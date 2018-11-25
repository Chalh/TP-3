import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

f = open("corpus/train.txt","r")

data_train = pd.read_csv(f,sep="\t")

X_train = []

#print (data_train)

for  x in data_train:
    sentence = " ".join(x[1:3])
    X_train.append(sentence)

print (X_train)


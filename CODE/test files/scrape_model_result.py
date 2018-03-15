# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:39:40 2017

@author: 212335848
"""
import os
import newspaper
import string
from itertools import combinations
from nltk.tag import pos_tag
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
#from textblob import TextBlob
import gensim
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import pickle

import Fakenews_featureextraction as fe

model = pickle.load(open("fakenews_model.dat", "rb"))


entered_url = 'http://www.cnn.com/2017/10/31/us/new-york-shots-fired/index.html' 
#entered_url = 'http://therealnews.com/t2/story:20601:Trump%27s-Friends-Get-Tax-Cuts%2C-His-Base-Gets-Bigotry'


article = newspaper.article.Article(url = entered_url)
article.download()
article.parse()

article.nlp()

#print(article.text)

text = article.text
'''
d = {'text': [text]}
text_df = pd.DataFrame(data=d)
externaltestinput = text_df
#externaltestinput=pd.read_csv('externaltestset.csv', encoding = 'latin-1')
#Clean the data and extract features
externaltest = fe.clean_text_add_features(externaltestinput)
#Vectorize the word
externaltest=fe.word_vectors(externaltest)

#Structuring the data
externaldata = externaltest[['specialchar','wordcount','avewordlength', 'firstpersonwordcount', 'uniquewords', 'capitalizedwords', 'vector']]
externaldata2=pd.concat([externaldata['vector'].apply(pd.Series), externaldata], axis = 1)
externaldata2.drop('vector', axis=1, inplace=True)
externaldata2.dropna(inplace=True)
externaldata_predictor = externaldata2.as_matrix()

xgb_externaltest_output = model.predict(externaldata_predictor)

if(xgb_externaltest_output):
    classification = "FAKE"
else:
    classification = "REAL"
    
print("-------------------------")
print("original df: ", externaltestinput)
#print("-------------------------")
#print("new tx df: ", text_df)
print("-------------------------")
print("Classification: ", classification)

#print(os.getcwd() + "\n")
#NLPmodel = gensim.models.KeyedVectors.load_word2vec_format(cur_dir+"/GoogleNews-vectors-negative300.bin", binary=True)
'''
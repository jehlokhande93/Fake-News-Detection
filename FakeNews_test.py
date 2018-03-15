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


cur_dir = os.getcwd()
os.chdir(cur_dir)


import Fakenews_featureextraction as fe

model = pickle.load(open("fakenews_model.dat", "rb"))


##External test set
#Load the data
externaltestinput=pd.read_csv('externaltestset.csv', encoding = 'latin-1')
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


print("-------------------------")
print("output: ")
print(xgb_externaltest_output)


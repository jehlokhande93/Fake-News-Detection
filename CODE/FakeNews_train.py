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


faketrain=pd.read_csv('dataset.csv', encoding='latin-1')

##CLEANING

datatrain = fe.clean_text_add_features(faketrain)

##VECTORIZER
NLPmodel = gensim.models.KeyedVectors.load_word2vec_format(cur_dir+"/GoogleNews-vectors-negative300.bin", binary=True)

datatrain=fe.word_vectors(datatrain)

##Subsetting and final structuring
finaldata = datatrain[['fake', 'specialchar','wordcount','avewordlength', 'firstpersonwordcount', 'uniquewords', 'capitalizedwords', 'vector']]

finaldata2=pd.concat([finaldata['vector'].apply(pd.Series), finaldata], axis = 1)
finaldata2.drop('vector', axis=1, inplace=True)

finaldata2.dropna(inplace=True)

#Random sampling of train data

finaldata_predictors = finaldata2.drop('fake',axis=1)
finaldata_response = finaldata2.as_matrix(['fake'])
finaldata_predictor = finaldata_predictors.as_matrix()

finaldatatrain_predictors, finaldatatest_predictors, finaldatatrain_response,  finaldatatest_response = train_test_split(finaldata_predictor, finaldata_response, test_size=0.2, random_state=40)

#SVM model - tested on SVM as a baseline
#svmmodel = svm.SVC()
#svmmodel.fit(finaldatatrain_predictors, finaldatatrain_response)
#
#svm_test_output = svmmodel.predict(finaldatatest_predictors)
#
#accuracy_score(finaldatatest_response, svm_test_output)


#SVM with SVD
#
#svd = TruncatedSVD(n_components=100)
#svd.fit(finaldatatrain_predictors)
#
#finaldatatrain_predictorssvd = svd.transform(finaldatatrain_predictors)
#finaldatatest_predictorssvd = svd.transform(finaldatatest_predictors)
#
#svmmodelsvd = svm.SVC()
#svmmodelsvd.fit(finaldatatrain_predictorssvd, finaldatatrain_response)
#
#svmsvd_test_output = svmmodelsvd.predict(finaldatatest_predictorssvd)
#
#accuracy_score(finaldatatest_response, svmsvd_test_output)



#XGBOOST
xgbmodel = xgb.XGBClassifier(nthread=10, silent=False, max_depth=7, learning_rate=0.2)
xgbmodel.fit(finaldatatrain_predictors, finaldatatrain_response)
xgb_test_output = xgbmodel.predict(finaldatatest_predictors)
accuracy_score(finaldatatest_response, xgb_test_output)

pickle.dump(xgbmodel, open("fakenews_model.dat", "wb"))






# Author: Taylor Million
from flask import render_template, flash, redirect
from flask import Flask, request
from pymongo import MongoClient
from app import app

import newspaper
import string
from itertools import combinations
from nltk.tag import pos_tag
from geotext import GeoText
from collections import Counter
from pytrends.request import TrendReq
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import gensim
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb

import pickle
from app import Fakenews_featureextraction as fe

@app.route("/")
def hello():
    return render_template("url_input.html")
 
@app.route("/results", methods=['POST', 'GET'])
def scrape():
   cur_dir = os.getcwd()
   os.chdir(cur_dir)
   if request.method == 'POST':
      MONGODB_HOST = 'localhost'
      MONGODB_PORT = 27017
      DBS_NAME = 'donorschoose'
      COLLECTION_NAME = 'projects'
      FIELDS = {}
      connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
      collection = connection[DBS_NAME][COLLECTION_NAME]
      projects = collection.find(projection=FIELDS, limit=5000)


      #Reading in the classification model
      model = pickle.load(open(cur_dir+"/fakenews_model.dat", "rb"))

      #Reading in the URL input by the user
      entered_url = request.form['text']
      
      #Creating an article object
      article = newspaper.article.Article(url = entered_url)
      article.download()
      article.parse()
      article.nlp()
      
      #Extracting the article text
      text = article.text
      d = {'text': [text]}
      text_df = pd.DataFrame(data=d)
      externaltestinput = text_df
      
      
      #Begin Classification of Article      
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
      
      SC = externaldata2['specialchar'][0]
      CW = externaldata2['capitalizedwords'][0]
      AWL = externaldata2['avewordlength'][0]
      
      awl_data = [100*np.abs(1-float(5/AWL)), 100*np.abs(float(5/AWL))]
      cw_data = [100*float(CW), 1000*(1-float(CW))]
      sc_data = [(1000*np.abs(SC)), 100-(1000*np.abs(SC))]
     
      #Running the model and storing the result
      xgb_externaltest_output = model.predict(externaldata_predictor)
      
      if(xgb_externaltest_output):
          classification = "FAKE"
      else:
          classification = "REAL" 
      # getting related words
      tagged_art = pos_tag(article.text.split())
      all_nouns = [word for word,pos in tagged_art if (pos == 'NN' or pos == 'NNP')]
      count = Counter(all_nouns)
      most_cmn = count.most_common()[0:5]
      
      noun_list = [None]*5
      for i in range(0,5):
          noun_list[i] = str(most_cmn[i][0])
      
      all_combos = list(combinations(noun_list,2))

      places = GeoText(article.text)
      all_cities = places.cities

      pytrend = TrendReq()
      #tf = 'now 7-d'
      tf = 'today 1-m'

      all_cities = list(set(all_cities))
      # checking if any combinations are cities

      prev_max = 0
      occur = dict(most_cmn)

      i = 0
      for item in all_combos:
          
          for city in all_cities:
              c1 = "" + item[0] + " " + item[1]
              c2 = "" + item[1] + " " + item[0]
              if(c1 in city  or c2 in city):
                  # remove both items, add back city
                  # and higher occurence item
                  # if equal, remove both
                  noun_list.remove(item[0])
                  noun_list.remove(item[1])
                  city_name = str(city).replace(" City","")
                  
                  
                  #if there are the same number of occurences
                  if(occur[item[0]] == occur[item[1]]):
                      noun_list.append(city_name)
                  
                  elif(occur[item[0]] > occur[item[1]]):
                      noun_list.append(city_name)
                      noun_list.append(item[0])
                  else:
                      noun_list.append(city_name)
                      noun_list.append(item[1])
                
          i = i + 1        
      
      final_combos = list(combinations(noun_list,2))  
      all_results = pd.DataFrame(columns = ['search_term', 'interest', 'related'],  index = [0,1,2,3,4,5,6,7,8,9])
      j = 0
      
      for item in final_combos:
          query = item[0] + " " + item[1]
          query = query.translate(string.punctuation)
          pytrend.build_payload(kw_list=[query], timeframe= tf)
          interest_over_time_df = pytrend.interest_over_time()
          
          related_queries_dict = pytrend.related_queries()
          
          # using the rising trends to observe the most popular search
          rise_query_df = related_queries_dict[query]['rising']
          all_results.iloc[j]['search_term'] = query
          all_results.iloc[j]['interest'] = interest_over_time_df
          all_results.iloc[j]['related'] = rise_query_df
      
          if(rise_query_df is not None):
              max_line = rise_query_df.loc[rise_query_df['value'].idxmax()]
              max_val = max_line['value']
              max_val_query = max_line['query']
              
              if(max_val > prev_max):
                  prev_max = max_val
                  prev_query = max_val_query
                  best = related_queries_dict
          j = j + 1
      
      pytrend.build_payload(kw_list=[prev_query], timeframe= tf)   
      interest_over_time_df = pytrend.interest_over_time()
      df = interest_over_time_df.drop(labels='isPartial', axis=1)


      py.iplot([{
          'x': df.index,
          'y': df[col],
          'name': col
      }  for col in df.columns], filename='simple-line')

   
      import networkx as nx

      G = nx.Graph()

          
      for i in range(all_results.shape[0]):
          n1 = all_results.iloc[i]
          G.add_node(n1['search_term'])
          if(i != 0):
              edge = (n1['search_term'], all_results.iloc[i-1]['search_term'])
              G.add_edge(*edge)
          #else:
          #    G.node[0]['pos'] = (10, 5)
      
          if(all_results.iloc[i]['related'] is not None):
              for j in range(n1['related'].shape[0]):
                  n2 = n1['related'].iloc[j]
                  G.add_node(n2['query'])
                  edge = (n1['search_term'], n2['query'])
                  G.add_edge(*edge)        
              
      pos = nx.spring_layout(G,k=2,iterations=20)
      nx.draw(G, node_size=1000, node_color='c', pos=pos, with_labels=True)
      #plt.savefig("simple_path.png")
      #plt.show()
      
      edge_trace = Scatter(
          x=[],
          y=[],
          line=Line(width=0.5,color='#888'),
          hoverinfo='none',
          mode='lines')
          
      node_trace = Scatter(
          x=[],
          y=[],
          text=list(G.nodes()),
          mode='markers',
          hoverinfo='text',
          marker=Marker(
              showscale=False,
              colorscale='YIGnBu',
              reversescale=True,
              color=[],
              size=20,
              colorbar=dict(
                  thickness=15,
                  title='Node Connections',
                  xanchor='left',
                  titleside='right'
              ),
              line=dict(width=2)))   
      
      
      
      for node in G.nodes():
          node_trace['x'].append(pos[node][0])
          node_trace['y'].append(pos[node][1])
          
      for edge in G.edges():
          x0, y0 = pos[edge[0]]
          x1, y1 = pos[edge[1]]
          edge_trace['x'] += [x0, x1, None]
          edge_trace['y'] += [y0, y1, None]    
          
      for node in enumerate(G.nodes()):
          node_trace['marker']['color'].append(0)
          node_info = str(node)
          node_trace['text'].append(node_info)
      
      fig = Figure(data=Data([edge_trace, node_trace]),
                   layout=Layout(
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      annotations=[ dict(
                          showarrow=False,
                          xref="paper", yref="paper",
                          x=0.005, y=-0.002 ) ],
                      xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

      py.iplot(fig, filename='networkx')
    
      #Update plotly plots based on the results from the model
      fig = {
      "data": [
        {
          "values": [],
          "hoverinfo": "none",
          "marker": {"colors": []},
          "textinfo": "none",
          "hole": 0.6,
          "type": "pie"
        }
      ],
      "layout": {
        "showlegend": False,
        "annotations": [
          {
            "text": "",
            "font": {"size": 10},
            "showarrow": False
          }
        ],
        "title": classification,
        "titlefont":{
                "family": "Courier New",
                "size": 180}, 
        "margin": {"t" : 600}
    
      }
    }     
      py.iplot(fig, filename='classification')
      
    #import plotly.plotly as py
    #from plotly.graph_objs import *
      trace1 = {
        "domain": {
          "x": [0, 0.31], 
          "y": [0.1, 1]
        }, 
        "hole": 0.6, 
        "hoverinfo": "none", 
        "labels": ["Data", ""], 
        "marker": {"colors": ["rgb(53, 196, 170)", "rgb(255, 255, 255)"], 
                   "line": {"color":["rgb(0, 0, 0)"], "width":2}
                              }, 
        "name": "CW", 
        "textinfo": "none", 
        "type": "pie", 
        "values": cw_data
      }
      trace2 = {
        "domain": {
          "x": [0.33, 0.64], 
          "y": [0.1, 1]
        }, 
        "hole": 0.6, 
        "hoverinfo": "none", 
        "labels": ["Data", ""], 
        "marker": {"colors": ["rgb(53, 196, 170)", "rgb(255, 255, 255)"], 
                   "line": {"color":["rgb(0, 0, 0)"], "width":2}
        }, 
        "name": "SC", 
        "textinfo": "none", 
        "type": "pie", 
        "values": sc_data
      }
      trace3 = {
        "domain": {
          "x": [0.66, 1], 
          "y": [0.1, 1]
        }, 
        "hole": 0.6, 
        "hoverinfo": "none", 
        "labels": ["Data", ""], 
        "marker": {"colors": ["rgb(53, 196, 170)", "rgb(255, 255, 255)"], 
                   "line": {"color":["rgb(0, 0, 0)"], "width":2}
        }, 
        "name": "AWL", 
        "textinfo": "none", 
        "type": "pie", 
        "values": awl_data
      }
      fig = {
              "data": [trace1, trace2, trace3],
             "layout" : {
        "annotations": [
          {
            "x": 0.1, 
            "y": 0.12, 
            "font": {"size": 16}, 
            "showarrow": False, 
            "text": "# Cap. Words"
          }, 
          {
            "x": 0.12, 
            "y": 0.04, 
            "font": {"size": 16}, 
            "showarrow": False, 
            "text": str(np.round(cw_data[0], 0))+"%"
          }, 
          {
            "x": 0.47, 
            "y": 0.12, 
            "font": {"size": 16}, 
            "showarrow": False, 
            "text": "# Special Char"
          }, 
          {
            "x": 0.46, 
            "y": 0.04, 
            "font": {"size": 16}, 
            "showarrow": False, 
            "text": str(np.round(sc_data[0], 0))+"%"
          }, 
          {
            "x": 0.9, 
            "y": 0.12, 
            "font": {"size": 16}, 
            "showarrow": False, 
            "text": "Avg. Word Len"
          }, 
          {
            "x": 0.85, 
            "y": 0.04, 
            "font": {"size": 16}, 
            "showarrow": False, 
            "text": str(np.round(awl_data[0], 0))+"%"
          }
        ], 
        "showlegend": False, 
        "title": "Score Component Breakdown"
      }
      }
      py.iplot(fig, filename='score_bd')
      
   connection.close()

   return render_template("output.html", article=article)
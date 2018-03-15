# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:39:40 2017

@author: 212335848
"""

import newspaper
import string
from itertools import combinations
from nltk.tag import pos_tag
from geotext import GeoText
from collections import Counter
from pytrends.request import TrendReq

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd


entered_url = 'http://www.cnn.com/2017/10/31/us/new-york-shots-fired/index.html' 

article = newspaper.article.Article(url = entered_url)
article.download()
article.parse()

article.nlp()

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

# testing out the first only   
#query = final_combos[0][0] + " " + final_combos[0][1]

for item in final_combos:
    query = item[0] + " " + item[1]
    query = query.translate(None, string.punctuation)
    #pytrend.build_payload(kw_list=[query], timeframe= tf)
    pytrend.build_payload(kw_list=[query], timeframe= tf)
    interest_over_time_df = pytrend.interest_over_time()
    
    related_queries_dict = pytrend.related_queries()
    
    # using the rising trends to observe the most popular search
    rise_query_df = related_queries_dict[query]['rising']

    if(rise_query_df is not None):
        max_line = rise_query_df.loc[rise_query_df['value'].idxmax()]
        max_val = max_line['value']
        max_val_query = max_line['query']
        
        if(max_val > prev_max):
            prev_max = max_val
            prev_query = max_val_query
            best = related_queries_dict

prev_query = 'new york attack'
pytrend.build_payload(kw_list=[prev_query], timeframe= tf)   
interest_over_time_df = pytrend.interest_over_time()
df = interest_over_time_df.drop(labels='isPartial', axis=1)
#df.plot()

py.iplot([{
    'x': df.index,
    'y': df[col],
    'name': col
}  for col in df.columns], filename='simple-line')

'''

data = [
    go.Scatter(
        x=df.index, # assign x as the dataframe column 'x'
        y=df.ix[:,0]
    )
]
'''
# IPython notebook
# py.iplot(data, filename='pandas/basic-line-plot')

#url = py.plot(data, filename='pandas/basic-line-plot')





# tried different combinations, to achieve desired result
# why trends worked with 2 nouns
# we google only on what we want results for
# googling the subject or object...these must be nouns
# subject balance between specificity and broadness
'''
# if city name contains "city", remove " City" from the text

#pytrend.build_payload(kw_list=[], timeframe= tf)


'''


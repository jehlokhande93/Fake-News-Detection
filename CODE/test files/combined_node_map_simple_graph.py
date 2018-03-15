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
import plotly.plotly as py
from plotly.graph_objs import *


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
        
    
    
    




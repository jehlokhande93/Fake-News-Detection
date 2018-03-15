# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:15:00 2017

@author: 212335848
"""

from pytrends.request import TrendReq
import plotly
import cufflinks as cf
import plotly.plotly as py

'''
df = cf.datagen.lines()

py.iplot([{
    'x': df.index,
    'y': df[col],
    'name': col
}  for col in df.columns], filename='simple-line')

'''

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()

# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
#pytrend.build_payload(kw_list=['Manhattan truck'])
pytrend.build_payload(kw_list=['Sophia execution'], timeframe='now 7-d')

# Interest Over Time
interest_over_time_df = pytrend.interest_over_time()
#print(interest_over_time_df.head())
print(interest_over_time_df)

# Related Queries, returns a dictionary of dataframes
related_queries_dict = pytrend.related_queries()
#print(related_queries_dict)

# Get Google Hot Trends data
trending_searches_df = pytrend.trending_searches()
print(trending_searches_df.head())


# Get Google Keyword Suggestions
suggestions_dict = pytrend.suggestions(keyword='pizza')
print(suggestions_dict)

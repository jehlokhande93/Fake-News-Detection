import plotly.plotly as py
import plotly.graph_objs as go





'''
# CLASSIFICATION RESULTS (added)
classification = "FAKE"
fig = {
  "data": [
    {
      "values": [],
      "hoverinfo": "none",
      "marker": {
        "colors": []
      },
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
        "font": {
          "size": 10
        },
      "showarrow": False
      }
    ],
    "title": classification,
    "titlefont":{
            "family": "Courier New",
            "size": 90}, 
    "margin": {"t" : 300}

  }
}
      
      
py.iplot(fig, filename='classification')
'''
'''
# SCORE COMPONENTS

import plotly.plotly as py
from plotly.graph_objs import *
trace1 = {
  "domain": {
    "x": [0, 0.31], 
    "y": [0.1, 1]
  }, 
  "hole": 0.6, 
  "hoverinfo": "none", 
  "labels": ["Data", ""], 
  "labelssrc": "tmillion3:9:0fb1bc", 
  "marker": {"colors": ["rgb(53, 196, 170)", "rgb(255, 255, 255)"]}, 
  "name": "Starry Night", 
  "textinfo": "none", 
  "type": "pie", 
  "values": [84, 16]
}
trace2 = {
  "domain": {
    "x": [0.33, 0.64], 
    "y": [0.1, 1]
  }, 
  "hole": 0.6, 
  "hoverinfo": "none", 
  "labels": ["Data", ""], 
  "labelssrc": "tmillion3:9:0fb1bc", 
  "marker": {"colors": ["rgb(53, 196, 170)", "rgb(255, 255, 255)"]}, 
  "name": "Sunflowers", 
  "textinfo": "none", 
  "type": "pie", 
  "values": [100, 0], 
  "valuessrc": "tmillion3:9:38922c"
}
trace3 = {
  "domain": {
    "x": [0.66, 1], 
    "y": [0.1, 1]
  }, 
  "hole": 0.6, 
  "hoverinfo": "none", 
  "labels": ["Data", ""], 
  "labelssrc": "tmillion3:9:0fb1bc", 
  "marker": {"colors": ["rgb(53, 196, 170)", "rgb(255, 255, 255)"]}, 
  "name": "Irises", 
  "textinfo": "none", 
  "type": "pie", 
  "values": [70, 30], 
  "valuessrc": "tmillion3:9:19fbb4"
}
fig = {
        "data": [trace1, trace2, trace3],
       "layout" : {
  "annotations": [
    {
      "x": 0.1, 
      "y": 0.25, 
      "font": {"size": 18}, 
      "showarrow": False, 
      "text": "# Cap. Words"
    }, 
    {
      "x": 0.12, 
      "y": 0.2, 
      "font": {"size": 18}, 
      "showarrow": False
    }, 
    {
      "x": 0.47, 
      "y": 0.25, 
      "font": {"size": 18}, 
      "showarrow": False, 
      "text": "# Special Char"
    }, 
    {
      "x": 0.46, 
      "y": 0.2, 
      "font": {"size": 18}, 
      "showarrow": False
    }, 
    {
      "x": 0.9, 
      "y": 0.23, 
      "font": {"size": 18}, 
      "showarrow": False, 
      "text": "Avg. Word Len"
    }, 
    {
      "x": 0.85, 
      "y": 0.2, 
      "font": {"size": 18}, 
      "showarrow": False
    }
  ], 
  "showlegend": False, 
  "title": "Score Component Breakdown"
}
}
py.iplot(fig, filename='score_bd')
'''

#

fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Network graph made with Python',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

py.iplot(fig, filename='networkx')




















'''

data_vals1 = [84, 16]
data_vals2 = [63, 15]
data_vals3 = [70, 30]

fig = {
  "data": [
    {
      "values": data_vals1,
      "labels": ["Data",""],
      "hoverinfo":"none",
      "hole": .6,
      "textinfo": "none",
      #"hoverinfo":'name',
      "marker": {"colors": ['rgb(53, 196, 170)', 'rgb(255, 255, 255)']},
      "type": "pie"
    }, 
    {
      "values": data_vals2,
      "labels": ["Data",""],
      "hoverinfo":"none",
      "hole": .6,
      "textinfo": "none",
      #"hoverinfo":'name',
      "marker": {"colors": ['rgb(53, 196, 170)', 'rgb(255, 255, 255)']},
      "type": "pie"
    }, 
    {
      "values": data_vals3,
      "labels": ["Data",""],
      "hoverinfo":"none",
      "hole": .6,
      "textinfo": "none",
      #"hoverinfo":'name',
      "marker": {"colors": ['rgb(53, 196, 170)', 'rgb(255, 255, 255)']},
      "type": "pie"
    }        ],
  "layout": {
        "title":"Score Component Breakdown",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": str(data_vals1[0])+"%"
            }
        ], 
        "showlegend": False
    }
}
#,"text": str(data_vals1[0])+"%"
py.iplot(fig, filename='donut_final')
'''


# One Plot

'''
data_vals = [85, 15]

fig = {
  "data": [
    {
      "values": data_vals,
      "labels": ["Data",""],
      "hoverinfo":"none",
      "hole": .6,
      "textinfo": "none",
      #"hoverinfo":'name',
      "marker": {"colors": ['rgb(255, 255, 255)', 'rgb(255, 255, 255)']},
      "type": "pie"
    }],
  "layout": {
        "title":"Title",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": ""
            }
        ], 
        "showlegend": False
    }
}
#import plotly plotly.tools.set_credentials_file(username='DemoAccount', api_key='lr1c37zw81')
py.iplot(fig, filename='donut')
'''
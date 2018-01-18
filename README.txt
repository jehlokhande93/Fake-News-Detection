CSE 6242 Data and Visual Analytics Final Project - Fake News Detection
# Author: Taylor Million, Jeh Lokhande

# 1. DESCRIPTION
### General Overview
This project uses Flask on the backend to host webpages and enable python script execution upon events observed on the webpage. 
For example, when the user types a URL into the input bar and hits enter, a script runs to scrape the text content of the entered URL. 
The results dashboard populates the fields of the keen.io dashboard with plotly plots that are generated during the execution of the classification.
### Interworkings
Flask is used on the backend to route web application traffic so that the article contents can be scraped and fed into the predefined model.
Flask renders templates from the \Harambe_Lives\demo\app\templates folder. Two html files are here. The initial Url Input Page (url_input.html)
and the results dashboard (output.html).

### Behind the Scenes
Flask is used to call a python script to do the following:
1. Scrape the article text using the newspaper API
2. Extract hand-crafted features along with additional features from the article text
3. Create a plot of the fakeness score according to selected had crafted features
4. Feed the article text into the trained model and record the result
5. Find top 5 most commonly used words in the article text and create all combinations pairs
6. Query GoogleTrends on every combination recording the following properties
	* query
	* interest over time
	* related queries
7. Generate a node mapping from the query-related queries, store the query with the highest interest over time and plot the result.
8. Render all of the generate plotly plots in the keen.io dashboard
 
-------------------------------------------------------------------------------------------------------------------------------------------------
# 2. INSTALLATION
Scraper Installation: http://newspaper.readthedocs.io/en/latest/
-----------------------------------
Folder Structure
-----------------------------------
Please ensure that the folder structure is not modified in any way. 
The scripts import/load objects saved in the folders.
------------------------------------
Python 3.6 Required
------------------------------------
Visualization Requirements
------------------------------------
pip install dash==0.18.3
pip install dash-renderer==0.10.0
pip install dash-html-components==0.7.0
pip install dash-core-components==0.12.6
pip install plotly --upgrade

pip install newspaper3k
pip install nltk

pip install Flask

#run
#import nltk
#nltk.download()

install mongodb: https://docs.mongodb.com/manual/installation/

pip install pytrends
pip install pymongo
pip install geotext
pip install re
pip install plotly

* Register for a plotly account and
* set up your configuration file with your corresponding
* username and API tokens

------------------------------------
Installation Requirements for Model
------------------------------------
pip install TextBlob
pip install gensim
pip install xgboost or try conda install -c mikesilva xgboost
pip install scipy
pip install sklearn
pip install pickle
*downgrade pandas to 0.20.3
------------------------------------
Downloads
------------------------------------
word2vec = https://github.com/mmihaltz/word2vec-GoogleNews-vectors
This file is 3.5GB and should be saved in the \Harambe_Lives\demo\app
directory as GoogleNews-vectors-negative300.bin
-------------------------------------------------------------------------------------------------------------------------------------------------

# 3. EXECUTION
Firstly, the model needs to be generated.
Navigate to the \Harambe_Lives\demo directory
Run python FakeNews_train.py
The result should be a file called fakenews_model.dat. 
This is the classification model.

Next, run MongoDB before opening the app
"C:\Program Files\MongoDB\Server\3.4\bin\mongod.exe"

Navigate to the app folder
run: python run.py

Type 127.0.0.1:4020 into web browser

Now, enter a valid URL in the format: http://webadress.topleveldomain
(Here is an example URL to try: http://www.cnn.com/2017/10/31/us/new-york-shots-fired/index.html)

*NOTE* Since we do not have enterprise plotly accounts and only personal, the configuration files could not be shared
meaning the links included in the output reference my personal plotly account and will not be modified when run elsewhere.
Classify an article, then navigate to your personal plotly, find the figures and replace them in the \Harambe_Lives\demo\app\templates\output.html file

Now, your files will be referenced and updated each time the package is run.





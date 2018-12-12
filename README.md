# Effects of Narrative Trajectories on the Popularities of Online News Articles
This repository holds the companion code for the project "Understanding the Effects of Narrative Trajectories on the Popularities of Online News Articles". 

Author: Raiyan Abdul Baten (rbaten@ur.rochester.edu)


## Installation Instructions
This program is created and tested in Python 3.6.3.

1. Install the basic requirements:
pip install -r requirements.txt

2. Install BeautifulSoup for scraping the mashable website:
pip install bs4

3. Install IBM Watson Developer Cloud to extract the sentence-wise tones:
pip install --upgrade watson-developer-cloud


## Usage Instructions
1. Article content scraping [OPTIONAL]:
The fetch_article_content.py file scrapes the mashable website to collect the article contents of the urls given in the original dataset. It saves all the files to the folder article_content_data. However, this operation is lengthy (takes more than 11 hours) and memory intensive (~90 MB). I have kept just the first two scraped articles in the folder as examples.

2. Fetching tone data from IBM Watson API [OPTIONAL]:
The fetch_bluemix_tones.py script sends the articles to the IBM Watson API, receives the tones, and saves them in pickle files in the folder bluemix_data. All the tone information files are included in this submission. To run the API call yourself and re-generate the tone information, you will first need to open an account in the IBM Watson Tone Analyzer website (https://www.ibm.com/watson/services/tone-analyzer/). You will be given an API key and a url when you create a service instance, which you will need to paste in the bluemix_key.py file. After that, you can run the fetch_bluemix_tones.py file to re-generate the tone information. This operation is also lengthy (takes more than 11.5 hours) and memory intensive (134 MB).

3. Analysis:
The file analysis.py loads the information from the pickle files and performs all the clustering operations and statistical tests. It prints the bonferroni corrected ttest results on the terminal.

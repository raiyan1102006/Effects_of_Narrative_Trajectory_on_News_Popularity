import pandas as pd
import os
import numpy as np
import re
import csv
import requests
from bs4 import BeautifulSoup

def load_csv_data(folder_name, file_name):
    csv_path = os.path.join(folder_name, file_name+".csv")
    return pd.read_csv(csv_path, encoding='latin-1')

def fetch_content(link):
    text=""
    page = requests.get(link)
    if(page.status_code==200):
        section = BeautifulSoup(page.content, 'html.parser').find('section', {'class': 'article-content'})
        if(not section):
            return "failed section NoneType"
        children = section.findChildren("p" , recursive=False)
        if(not children):
            return "failed children NoneType"
        for child in children[:-1]: #usually last entry holds credits info
            string = re.sub('<a.*?>|</a>', '', str(child)[3:-4]) # [3:-4] gets rid of the <p></p> tags
            string = re.sub('<em.*?>|</em>', '', string)
            string = re.sub('<strong.*?>|</strong>', '', string)
            string = re.sub('<div.*?>|</div>', '', string)
            string = re.sub('<span.*?>|</span>', '', string)
            string = re.sub('<i.*?>|</i>', '', string)
            string = re.sub('<img.*?/>', '', string)
            string = re.sub('<object.*?>|</object>', '', string)
            string = re.sub('<param.*?/>', '', string)
            string = re.sub('<iframe.*?>|</iframe>', '', string)
            if(string[:8]=="SEE ALSO"): #get rid of ad links
                continue
            text+=" ".join(string.split()) #remove redundant spaces
            text+=" " #make sure each sentence has a space after it
    else:
        return "failed"
    return text


if __name__ == "__main__":
	df_data = load_csv_data("","onlinenewspopularity")
	for index, row in df_data[:2].iterrows(): #change to df_data.iterrows() to generate complete article collection
	    print("Fetching article #"+str(index)+": "+row["url"])
	    file = open("article_content_data/"+str(index)+".txt","w") 
	    file.write(fetch_content(row["url"])) 
	    file.close() 





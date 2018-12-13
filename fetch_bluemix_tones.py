import pandas as pd
import os
import numpy as np
import re
import csv
import json
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from bluemix_key import *

#loads csv data
def load_csv_data(folder_name, file_name):
    csv_path = os.path.join(folder_name, file_name+".csv")
    return pd.read_csv(csv_path, encoding='latin-1')


# This function makes the calls to IBM Watson API, receives the tone data, and saves to pickle files
def fetch_bluemix_data(dataframe,start_idx,end_idx):

	for index, row in dataframe[start_idx:end_idx].iterrows():
	    print("Fetching tone of article #"+str(index)+": "+row["url"])
	    
	    #open file and load text
	    text_file = open('article_content_data/'+str(index)+'.txt', 'r')
	    text=text_file.read()
	    text_file.close()
	    
	    #check for minimum length. Discard articles with 5 sentences or less
	    number_of_sentences = sent_tokenize(text)
	    if(len(number_of_sentences)<6):
	        df_data.at[index, 'valid_article'] = 0
	        print("skipped article #"+str(index))
	        continue
	    
	    # fetch tone information from bluemix
	    temp_count=0
	    fail_flag=0
	    while True: #sometimes the API crashes. This loop tries for upto 5 times to get a valid API response
	        try:
	            tone_analysis = tone_analyzer.tone({'text': text},'application/json').get_result()
	        except:
	            temp_count+=1
	            if(temp_count==5):
	                df_data.at[index, 'valid_article'] = 0
	                print("failed repeatatively with article #"+ str(index)+". Please check your credentials!")
	                fail_flag=1
	                break
	            print("trying again, try "+str(temp_count))
	            continue
	        break
	    if(fail_flag):
	    	continue
	    
	    #save info into pickle file
	    bm_dump_file = open('bluemix_data/'+str(index)+'.pkl', 'wb')
	    pickle.dump(tone_analysis, bm_dump_file)
	    bm_dump_file.close()


if __name__ == "__main__":
	df_data = load_csv_data("","onlinenewspopularity")
	df_data['valid_article'] = 1
	fetch_bluemix_data(df_data,start_idx=0,end_idx=2) #change end_idx range to 39644 run the analysis on all the articles
	df_data.to_csv('df_data_new.csv', index=False)

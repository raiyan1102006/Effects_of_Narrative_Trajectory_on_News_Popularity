import pandas as pd
import os
import numpy as np
import re
import csv
import json
import pickle
import nltk
nltk.download('punkt')
import matplotlib
import matplotlib.pyplot as plt
from list_of_articles import all_valid_articles
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import f_oneway,ttest_ind
import itertools
import operator as op
from functools import reduce
from collections import OrderedDict

#loads csv data
def load_csv_data(folder_name, file_name):
    csv_path = os.path.join(folder_name, file_name+".csv")
    return pd.read_csv(csv_path)

#helps with combination calculation 
def count_n_choose_r(n,r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom

# this function helps in drawing boxplots
def draw_boxplots(pvals,allvals,s,comparator,outfilename=None):
    for akw in pvals:
        plt.figure(comparator.column_names[s]+akw)
        ax=plt.boxplot(allvals[akw],
            labels=comparator.groups.keys(),
            showfliers=False)
        plt.suptitle(\
            'Significant (p={0:0.6f}) difference in '.format(pvals[akw])+\
            akw+'\n'+'while clustering based on: '+comparator.column_names[s])
        if not outfilename:
            plt.show()
        else:
            plt.savefig(outfilename+'boxplt_'+\
                comparator.column_names[s]+'_'+akw+'.eps')
            plt.close()


def decorate_axis(c,cols,rows,yval,avg_yval,txtlist,legendval,fig,
        toff=0.03,boff=0.015,loff=0.02,midoff=0.03,roff=0.005,txth=0.18):
    irow = c / cols
    icol = c % cols
    cellw = (1. - loff - roff)/float(cols)
    cellh = (1. - toff - boff)/float(rows)
    axh = (cellh-midoff)/2.
    axleft = loff+icol*cellw+midoff/2.
    axbottom = boff+irow*cellh+midoff/2.+axh
    axw = cellw-midoff
    txtaxbottom = boff+irow*cellh+midoff/2.
    # Position the axes
    ax = fig.add_axes([axleft,axbottom,axw,axh])
    
    ax.plot(yval,color='gray',linewidth=0.5)
    ax.plot(avg_yval,color='orange',\
        linewidth=2,label=legendval)
    plt.ylim([0,1])
    plt.xlabel('Percent of Speech')
    plt.ylabel('Value')
    plt.legend()
    # Put the text axis
    txtax = fig.add_axes([axleft,txtaxbottom,axw,abs(axh-toff)])
    txtax.axis('off')
    txtax.patch.set_alpha(0)
    for i,txt in enumerate(txtlist):
        txtax.text(0,1 - txth*(i+1),str(i+1)+'. '+txt)

#Draws the cluster means and its closest-matching talks.

def draw_clusters_pretty(avg_dict,comp,csvcontent,vid_idx,
    b_=None,outfilename=None):

    X = np.array([comp.sentiments_interp[an_article] for an_article in comp.allarticles])
    M = np.size(X,axis=1)
    colidx = {col:i for i,col in enumerate(comp.column_names)}
    kwlist = ['shares']
    for ascore in avg_dict:
        # b is the index of the current score
        b = colidx[ascore]
        # If b_ is specified, just draw one score and skip others
        if b_ and not b_ == b:
            continue
        # Start plotting
        fig = plt.figure(figsize=(15,7))
        nb_clust = len(avg_dict[ascore].keys())
        rows = int(np.ceil(nb_clust/3.))
        cols = 3
        print(" ")
        print('######################')
        print("Tone: "+str(ascore))
        print('######################')
        print(" ")
        print("Generated clusters:")
        for c,aclust in enumerate(avg_dict[ascore]):
            # Standerdize X
            xmean = np.mean(X[:,:,b],axis=1)[None].T
            xstd = np.std(X[:,:,b],axis=1)[None].T
            Z = (X[:,:,b] - xmean)
            # Calculate the closest matches
            r = Z - avg_dict[ascore][aclust][None]
            simidx=np.argsort(np.sum(r*r,axis=1))
            yval = X[simidx[:20],:,b].T
            avg_yval = avg_dict[ascore][aclust]
            # Make the text to be shown for each cluster            
            txtlist = [csvcontent['title'][vid_idx[comp.allarticles[idx]]]\
                for idx in simidx[:5]]
            # Print the rating averages of the clusters
            f20vids=[vid_idx[comp.allarticles[idx]] for idx in simidx[:20]]
            print(aclust)
            

            # Draw the axes
            decorate_axis(c,cols,rows,yval,avg_yval,txtlist,aclust,fig)
        plt.suptitle(ascore.replace('_',' '))
        if not outfilename:
            plt.show()
        else:
            plt.savefig(outfilename+'clust_'+ascore+'.eps')
            plt.close()

# performs clustering assuming there is only one sentiment score
def clust_onescore_stand(X_1,clusterer,comparator):
    
    result_dict = {}
    mean_ = np.mean(X_1,axis=1)[None].T
    std_ = np.std(X_1,axis=1)[None].T
    Z = (X_1-mean_)
    clusterer.fit(Z) #DBScan clusterer
    labls = clusterer.labels_
    for lab,articleid in zip(labls,comparator.allarticles):
        if result_dict.get('cluster_'+str(lab)):
            result_dict['cluster_'+str(lab)].append(articleid)
        else:
            result_dict['cluster_'+str(lab)]=[articleid]
    return result_dict

# draws the cluster means and evaluate the differences in various clusters.
# Then performs airwise multiple t-test with Bonferroni correction

def evaluate_clust_separate_stand(X,clusterer,comparator,\
    csvcontent,csv_id,b_=None,outfilename=None):
    N,M,B = X.shape
    avg_dict = {}
    kwlist = ['shares']
    plt.close('all')
    # s is the index of a bluemix score
    for s in range(B):
        # If b_ is specified, just compute one score and skip others
        if b_ and not b_ == s:
            continue
        # Perform clustering over each score
        clust_dict = clust_onescore_stand(X[:,:,s],clusterer,comparator)
        comparator.reform_groups(clust_dict)
        avg = comparator.calc_group_mean()
        for aclust in avg:
            if not comparator.column_names[s] in avg_dict:
                avg_dict[comparator.column_names[s]] = {aclust:avg[aclust][:,s]}
            else:
                avg_dict[comparator.column_names[s]][aclust]=avg[aclust][:,s]
        # Pretty draw the clusters
        draw_clusters_pretty(avg_dict,comparator,csvcontent,csv_id,
            b_=s,outfilename=outfilename)
        if(len(clust_dict)<2):
            continue
        
        # Now apply ANOVA and compare clusters
        pvals = {}
        allvals = {}
        # Formulate a list of values for each rating
        print(" ")
        print('='*50)
        print('{:^50}'.format('HYPOTHESIS TESTS'))
        print('{:^50}'.format('for IBM Score:'+comparator.column_names[s]))
        print('='*50)
        for akw in kwlist:
            ratvals = {aclust:[int(csvcontent[akw][csv_id[avid]]) for avid\
                in comparator.groups[aclust]] for aclust in \
                comparator.groups}

            #################### perform ANOVA #####################
            ratval_itemlist = list(zip(*ratvals.items()))[1]
            _,pval = f_oneway(*ratval_itemlist)
            # Save only the statistically significant ones
            if pval<0.05:
                print('ANOVA p value ('+akw+'):',pval)
                # Bonferroni Correction for tests over multiple ratings
                print('ANOVA p value ('+akw+') with Bonferroni:',\
                    pval*float(len(kwlist)))
                if pval*float(len(kwlist)) < 0.05:
                    print('< 0.05')
                    pvals[akw]=pval*float(len(kwlist))
                    allvals[akw] = ratval_itemlist
                else:
                    print('not significant')
            ########### Pair-wise t-test with correction ###########
           
            # Total number of repeated comparisons
            paircount = count_n_choose_r(len(ratvals),2)
            # Pair-wise comparison using t-test and effectsize
            for rat1,rat2 in itertools.combinations(ratvals,2):
                _,pval_t = ttest_ind(ratvals[rat1],ratvals[rat2],\
                    equal_var=False)
                # Perform Bonferroni Correction for multiple t-tests
                # and multiple ratings
                pval_t = pval_t*float(paircount)*float(len(kwlist))
                # Check significance
                if pval_t < 0.05:
                    print('p-val of ttest (with Bonferroni) in "'+akw+\
                        '" between '+rat1+' and '+rat2+':',pval_t)
                    ############# Pair-wise Effectsizes ##############
                    n1 = len(ratvals[rat1])
                    n2 = len(ratvals[rat2])
                    sd1 = np.std(ratvals[rat1])
                    sd2 = np.std(ratvals[rat2])
                    sd_pooled = np.sqrt(((n1 - 1)*(sd1**2.) +\
                        (n2-1)*(sd2**2.))/(n1+n2-2))
                    cohen_d = (np.mean(ratvals[rat1]) - \
                        np.mean(ratvals[rat2]))/sd_pooled
                    print('Cohen\'s d of rating "'+akw+'" : ',cohen_d)
        # If the clusters are significantly different in any rating, draw it
        if not pvals.keys():
            continue
        else:
            draw_boxplots(pvals,allvals,s,comparator,outfilename=outfilename)



def read_data(dataframe):
    # Read the content of the index file
    # content is a dictionary 
    i=0
    vid_idx={}
    content={}
    for index,arow in dataframe.iterrows():
        vid_idx[index]=i
        if not content.get('title'):
            content['title']=[arow['url']]
        else:
            content['title'].append(arow['url']) 

        if not content.get('shares'):
            content['shares']=[int(arow[' shares'])]
        else:
            content['shares'].append(int(arow[' shares'])) 
        i+=1      
    return content,vid_idx

# Draw the cluster means and evaluate the differences in various clusters

def evaluate_clusters(X,comp,dataframe,outfilename='./plots/'):
    
    km = DBSCAN(eps=.25) #DBScan initialize
    content,vid_idx = read_data(dataframe)
    evaluate_clust_separate_stand(X,km,comp,content,vid_idx,outfilename=outfilename)


# helps parsing the json sent from IBM
def parse_tone_categories(tones_list):
    #these are the various tones
    header=['anger', 'disgust', 'fear', 'joy', 'sadness', \
            'analytical', 'confident', 'tentative', \
            'openness_big5', 'conscientiousness_big5', 'extraversion_big5', 'agreeableness_big5', 'emotional_range_big5']
    scores=len(header)*[0]
    for atone in tones_list:
        temp_index = header.index(atone["tone_id"])
        scores[temp_index]=atone["score"]
    return header,scores

#parses through the json sent from IBM
def parse_sentence_tone(senttone_list):
    sentences=[]
    header=[]
    scores=[]
    for asent in senttone_list:
        sentences.append(asent['text'])
        if asent['sentence_id']==0:
            header,score = parse_tone_categories(asent['tones'])
        else:
            _,score = parse_tone_categories(asent['tones'])
        if not score:
            continue
        scores.append(score)
    scores = np.array(scores)
    return scores,header,sentences


# Reads all the sentences and their corresponding bluemix sentiments.
def read_bluemix(pklfile,sentiment_dir='./bluemix_data/'):
  
    pklfile = sentiment_dir+pklfile.split('/')[-1]
    #make sure file is there
    assert os.path.isfile(pklfile),'File not found: '+pklfile
    assert os.path.isfile(pklfile),'Sentiment file not found: '+pklfile+\
        ' \nCheck the sentiment_dir argument'
    data = pickle.load(open(pklfile, 'rb'))
    assert data.get('sentences_tone'), \
        'Sentence-wise sentiment is not available: {0}'.format(pklfile)
    scores,header,sentences = parse_sentence_tone(data['sentences_tone'])
    return scores,header,sentences


# class holds all the loaded information in self for easy navigation
class Sentiment_Comparator(object):
    def __init__(self,
                dict_groups,
                reader=read_bluemix,
                inputFolder='./bluemix_data/',
                process=True):
        self.inputpath=inputFolder
        self.reader = reader    
        self.groups = dict_groups
        self.allarticles = [ids for agroup in self.groups for ids in self.groups[agroup]]
        self.raw_sentiments = {}
        self.sentiments_interp={}
        self.back_ref={}
        self.column_names=[]
        if process:
            self.extract_raw_sentiment() #loads the tone information
            self.smoothen_raw_sentiment() #smoothens them
            self.intep_sentiment_series() #interpolates them
    
    def extract_raw_sentiment(self):
        for index,an_article in enumerate(self.allarticles):
            filename = self.inputpath+str(an_article)+'.pkl' #the pickle file containing the tone info
            scores,header,_ = self.reader(filename)
            if index==0:
                self.column_names = header
            self.raw_sentiments[an_article] = scores #raw scores. This will be processed
    
    def smoothen_raw_sentiment(self,kernelLen=2): #using a kernel length of 2
        # Get number of columns in sentiment matrix 
        _,n = np.shape(self.raw_sentiments[self.allarticles[0]])

        for an_article in self.allarticles:
            temp=[]
            for i in range(n):
                temp.append(np.convolve(\
                self.raw_sentiments[an_article][:,i],\
                np.ones(kernelLen)/float(kernelLen),mode='valid'))
            self.raw_sentiments[an_article]=np.array(temp).T

    def intep_sentiment_series(self,bins=10): #interpolating to a canonical length of 10
        '''
        Fills out the variable self.sentiments_interp. Different sentiment
        series has different lengths due to the variable length of the articles.
        This function brings all the series in a common length (having 
        10 samples).
        '''        
        for an_article in self.allarticles:
            m,n = np.shape(self.raw_sentiments[an_article])
            # Pre-allocate
            self.sentiments_interp[an_article] = np.zeros((bins,n))
            # x values for the interpolation
            old_xvals = np.arange(m)
            new_xvals = np.linspace(0,old_xvals[-1],num=bins)
            # Update the backward reference
            self.back_ref[an_article] = [np.where((old_xvals>=lo) & \
                (old_xvals<=hi))[0].tolist() for lo,hi in \
                zip(new_xvals[:-1],new_xvals[1:])]+[[old_xvals[-1]]]
            # Interpolate column by column
            for i in range(n):
                self.sentiments_interp[an_article][:,i] = \
                np.interp(new_xvals,old_xvals,self.raw_sentiments[an_article][:,i])
                
    def reform_groups(self,new_dict_groups):
        
        self.groups = new_dict_groups
        self.raw_sentiments = {an_article:self.raw_sentiments[an_article] \
            for akey in new_dict_groups \
                for an_article in new_dict_groups[akey]}
        self.sentiments_interp = {an_article:self.sentiments_interp[an_article] \
            for akey in new_dict_groups \
                for an_article in new_dict_groups[akey] }
        
    def calc_group_mean(self):
        group_average = {}
        for agroup in self.groups:
            vals = [self.sentiments_interp[id] for id in self.groups[agroup]]
            # Averaging over the talks in a group
            group_average[agroup]=np.mean(vals,axis=0)
        return group_average

def load_all_scores():
    m = len(all_valid_articles)
    dict_input = {'all':all_valid_articles}
    # Load into sentiment comparator for all the pre-comps
    comp = Sentiment_Comparator(dict_input,read_bluemix)
    X = np.array([comp.sentiments_interp[an_article] for an_article in comp.allarticles])
    return X,comp


if __name__ == "__main__":
	df_data = load_csv_data("","df_data_full")
	df_data2 = df_data[df_data.valid_article == 1]

	print("###### Loading data from pickle files ######")
	X,comp = load_all_scores()

	print("###### Performing cluster analysis ######")
	evaluate_clusters(X,comp,df_data2,outfilename='./plots/')
	print("###### Please check the plot folder for the results ######")
#!/usr/bin/env python
# coding: utf-8

# # Task: 
# ## To Predict the Ratings using BoardGamesGeek-Reviews dataset
# ## Author: Rohit Singh ~~ ID:1001718350

# Intoduction:
# 
# Use the board game geek review dataset
# https://www.kaggle.com/jvanelteren/boardgamegeek-reviews 
# 
# 
# Here we have few types of classification algorithms in machine learning that we used to implement our project:
#  - Logistic Regression
#  - Naive Bayes Classifier
#  - Support Vector Machines
#  - Random Forest
# 
# Logistic Regression (Predictive Learning Model) :
# It is a statistical method for analyzing a data set in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). The goal of logistic regression is to find the best fitting model to describe the relationship between the dichotomous characteristic of interest (dependent variable = response or outcome variable) and a set of independent (predictor or explanatory) variables. This is better than other binary classification like nearest neighbor since it also explains quantitatively the factors that lead to classification.
# 
# Naive Bayes Classifier (Generative Learning Model) :
# It is a classification technique based on Bayes’ Theorem with the assumption of independence among predictors. In other words , a Naive Bayes classifiers assume that the presence of a particular feature in a class is unrelated to the presence of any other feature or that all of these properties have independent contribution to the probability. This family of classifiers is relatively easy to build and particularly useful for very large data sets as it is highly scalable. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.
# 
# Support-vector machines :
# These are supervised learning models with associated learning algorithms which analyze the data used for classification and regression analysis. In the light of a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm creates a model which assigns new examples to one or the other category. 
# 
# Random Forest:
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees’ habit of over fitting to their training set.

# In[1]:



import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import string


# storing the bgg-13m-reviews.csv as a dataframe

# In[2]:


data  = pd.read_csv("bgg-13m-reviews.csv")


# # Preprocessing Data ~Feature Selection
# 
# Only select Features that most apply to your predictive variable or performance you 're interested in. With irrelevant features in your data, the model's accuracy can be reduced drastically. so, inorder to do that we,
# 
#  - purge all rows containing null values in "comment" coloumn

# before :

# In[3]:


data=data.drop(["Unnamed: 0"], axis = 1)
data.head(5)


# In[4]:


print("Total No of reviews ",data.shape[0] )


# In[5]:


df = data.dropna()


# after :

# In[6]:


data=df.reset_index()
data=data.drop(["index"], axis = 1)
data.head(5)


# In[7]:


decimals = pd.Series([0], index=['rating'])
data = data.round(decimals)
#Ratings wise distribution
data["rating"].value_counts()


# # Checking for Imbalance in dataset with respect to number of ratings 
#  - it is alwasy wise to sample out unbiased data before training

# In[8]:


plot_imbal_labels = data.groupby(["rating"]).size()
plot_imbal_labels = plot_imbal_labels / plot_imbal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_imbal_labels.keys(), plot_imbal_labels.values).set_title("Before Down Sampling")
ax.set_ylabel('Number of samples')


# In[9]:


import seaborn as sns
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

data['text length'] = data.comment.apply(len)


# # Down sampling the imbalanced data
# since we only have 20000 data samples for the user rating 1.0 in the whole dataset, we took random sample of 20k from every other rating to maintain the bias which can negatively impact the overall accuracy for predicting ratings accordingly

# In[10]:


stars_10 = data[data["rating"]==10.0]
print(stars_10.shape)
stars_10_downsample = resample(stars_10, replace=False, n_samples=20000, random_state=0)
stars_10_downsample.shape


# In[11]:


stars_9 = data[data["rating"]==9.0]
print(stars_9.shape)
stars_9_downsample = resample(stars_9, replace=False, n_samples=20000, random_state=0)
stars_9_downsample.shape


# In[12]:


stars_8 = data[data["rating"]==8.0]
print(stars_8.shape)
stars_8_downsample = resample(stars_8, replace=False, n_samples=20000, random_state=0)
stars_8_downsample.shape


# In[13]:


stars_7 = data[data["rating"]==7.0]
print(stars_7.shape)
stars_7_downsample = resample(stars_7, replace=False, n_samples=20000, random_state=0)
stars_7_downsample.shape


# In[14]:


stars_6 = data[data["rating"]==6.0]
print(stars_6.shape)
stars_6_downsample = resample(stars_6, replace=False, n_samples=20000, random_state=0)
stars_6_downsample.shape


# In[15]:


stars_5 = data[data["rating"]==5.0]
print(stars_5.shape)
stars_5_downsample = resample(stars_5, replace=False, n_samples=20000, random_state=0)
stars_5_downsample.shape


# In[16]:


stars_4 = data[data["rating"]==4.0]
print(stars_4.shape)
stars_4_downsample = resample(stars_4, replace=False, n_samples=20000, random_state=0)
stars_4_downsample.shape


# In[17]:


stars_3 = data[data["rating"]==3.0]
print(stars_5.shape)
stars_3_downsample = resample(stars_3, replace=False, n_samples=20000, random_state=0)
stars_3_downsample.shape


# In[18]:


stars_2 = data[data["rating"]==2.0]
print(stars_2.shape)
stars_2_downsample = resample(stars_2, replace=False, n_samples=20000, random_state=0)
stars_2_downsample.shape


# In[19]:


stars_1 = data[data["rating"]==1.0]
print(stars_1.shape)
stars_1_downsample = resample(stars_1, replace=False, n_samples=20000, random_state=0)
stars_1_downsample.shape


# In[20]:


#Concatening the samples and defining the dataframe
reviews_ds = pd.concat([stars_1_downsample,stars_2_downsample,stars_3_downsample,stars_4_downsample,stars_5_downsample,stars_6_downsample,stars_7_downsample,stars_8_downsample,stars_9_downsample,stars_10_downsample])


# In[21]:


reviews_ds.shape


# In[22]:


reviews_ds['rating'].value_counts()


# In[23]:


plot_bal_labels = reviews_ds.groupby(["rating"]).size()
plot_bal_labels = plot_bal_labels / plot_bal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_bal_labels.keys(), plot_bal_labels.values).set_title("After Down Sampling")
ax.set_ylabel('Proportion of samples')


# In[24]:


reviews_ds['comment']


# # Contribution1: Clear text from jargon words for better accuraccy
# One of the most significiant contribution which helped to estimate relavent words for modeling and predict much accurate ratings

# In[25]:


d = {'whom', "you're", 'but', 'at', 'more', 'themselves', 'couldn', 'why', 'few', "you've", 'your',
        'doesn', 'before', 'him', 'she', 'don', 'what', 'are', 'doing', 'theirs', 'all', 've', 'into', 'player',
        'himself', 'same', 'has', 'above', 'was', 'where', "wasn't", 'the', 'just', 'again', 'm', 'isn',  'de',
        'did', 'does', 'or', "won't", 'yourself', 'here', 'll', 'through', 'because', 'about', 'which', 'watching',
        'ours', 'herself', 'my', 'this', 'how', 'during', 'until', 'between', 'aren', 'ma', 'be', 'of', 'going',
        'some', 'wasn', 'too', "didn't", "isn't", "doesn't", 'then', 'itself', "mightn't", 'd', 'from',
        'them', 'can', 'each', 'no', 'ourselves', 'won', 'ain', 'yours', 'myself', 'such', 'to', "needn't", 
         'mustn', "should've", 'when', 'weren', 'shouldn', 'haven', "mustn't", 'in', 'we', 'down', 'me',
        'against', 'both', 'needn', 'those', 'an', 'only', "that'll", 'hers', 'with', 'by', 'will', 'people',
        'wouldn', 'had', 'while', 'out', "you'll", 'not', 't', "weren't", "hadn't", 'hadn', 'if', 's', 'bgg',
        "hasn't", 'mightn', 'any', 'their', 'were', 'having', 'now', 'hasn', 'it', 'so', 'its', 'he',
        'y', 'should', 'you', 'me', 'yourselves', 'a', 'off', "haven't", "you'd", 'i', 'our', 'who', "shan't", 
         'further', 'is', 'very', 'her', "shouldn't", 'am', 'o', 're', 'over', 'shan', 'once', 'for', 'also',
        'been', 'there', 'own', "shan't", 'on', 'down', 'do', "wouldn't", 'his', 'these', 'most', 'that',
        "it's", 'and', 'nor', "don't", 'other', "she's", 'after', 'below', 'didn', "aren't", 'they',
        'being', "couldn't", 'have', 'than', 'up', 'as', 'under', 'film','movie','one','time','see','story','well',
         'like','even','good','also''first','get','much','first','point', 'box', 'others', 'mechanic', 'felt', 'tile',
        'plot','films','many','movies','made','acting','thing','way','think','character','did', 'such', 'doing', 
       'just', 'very', 'shan', 'against', 't', "you're", 'who', 'than', 'br','music','however','must','take','big',
          'only', "haven't", 'yours', 'you', 'its', 'other', 'we', 'where', 'then', 'they', 'won', "you've",
          'some', 've', 'y', 'each', "you'll", 'them', 'to', 'was', 'once', 'and', 'ain', 'under', 'through',
          'for', "won't", 'mustn', 'a', 'are', 'that', 'at', 'why', 'any', 'nor', 'these', 'yourselves', 'board',
          'has', 'here', "needn't", 'm', 'above', 'up', 'more', 'if', 'ma', 'didn', 'whom', 'can', 'have',
          'an', 'should', 'there', 'couldn', 'her', 'how', 'of', 'doesn', "shouldn't", 'further', 'rule',
          "wasn't", 'between', 'd', 'wouldn', 'his', 'being', 'do', 'when', 'hasn', "she's", 'by', "should've",
          'into', 'aren', 'weren', 'as', 'needn', 'what', "it's", 'hadn', 'with', 'after', 'he', 'off', 'not',
          'does', 'own', "weren't", "isn't", 'my', 'too', "wouldn't", 'been', 'again', 'same', 'few', "don't",
          'our', 'myself', 'your', 'before', 'about', 'most', 'during', 'll', 'on', 'shouldn', 'is', 'out',
         'below', 'which', 'from', 'she', 'were', 'those', 'over', 'until', 'theirs', 'mightn', 'random', 'nostar',
          'yourself', 'i', 'am', 'so', 'himself', 'it', 'had', 'or', 'all', 'while', "aren't", 'ours', 'strategy',
          "that'll", 'but', 'because', 'in', 'now', 'themselves', 'him', "doesn't", 'both', 're', 'wasn', 'theme', 
          's', "hasn't", "didn't", 'their', "mustn't", 'herself', 'the', 'this', 'will', 'isn', "you'd", 'game', 
          'haven', 'itself', "couldn't", 'o', 'be', 'don', 'hers', "mightn't", 'having', "hadn't", 'ourselves',
        'characters','watch','could','would','really','two','man','show','seen','still','never','make','little',
        'life','years','know','say','end','ever','scene','real','back','though','world','go','new','something',
       'scenes','nothing','makes','work','young','old','find','us','funny','actually','another','actors','director',
       'series','quite','cast','part','always','lot','look','love','horror','want','minutes','pretty','better','great',
       'best','family','may','role','every', 'performance','bad','things','times','bad','great','best','script','every',
       'seems','least','enough','original','action','bit','comedy','saw', 'long','right','fun','fact','around','guy', 'got',
       'anything','point', 'give','thought', 'whole', 'gets', 'making','without','day', 'feel', 'come','played','almost',
      'might', 'money', 'far', 'without', 'come', 'almost','kind', 'done','especially', 'yet', 'last', 'since', 'different',
       'although','true','interesting', 'reason', 'looks', 'done', 'someone', 'trying','job', 'shows', 'woman', 'tv', 
       'probably', 'father', 'girl', 'plays', 'instead', 'away', 'girl', 'probably', 'believe', 'sure', 'course', 'john', 
       'rather', 'later', 'dvd', 'war', 'found', 'looking', 'anyone', 'maybe', 'rather', 'let', 'screen', 'year', 'hard', 
       'together', 'set', 'place','comes', 'half', 'idea', 'american', 'play', 'takes', 'performances', 'everyone','actor',
     'wife', 'goes','sense', 'book', 'ending', 'version', 'star', 'everything', 'put', 'said', 'seeing', 'house', 'main'
     , 'three' ,'watched', 'high', 'else', 'men', 'night','need', 'try', 'kid','prefer', 'group', 'system', 'game','card'
        ,'playing','turn','dice','roll','move','hour','draw', 'le', 'deck', 'component', 'gameplay', 'choice', 'design'
        ,'size', 'hand', 'number', 'add', 'keep', 'chance', 'add', 'ok', 'gave', 'round', 'win', 'decision', 'experience'
         ,'oh','used','type','basically','next', 'update', 'url', 'seem', 'building','child','completely','either',
         'simply','easy', 'second', 'rolling','opponent', 'start','guess','space', 'understand','tried', 'artwork',
         'quality', 'randomness', 'scoring', 'map', 'element', 'remember', 'bought', 'gaming', 'totally','light','rating',
         'nice', 'piece', 'buy', 'art', 'use', 'wanted','combat','based', 'read', 'word', 'order','use', 'getting','control'
        ,'monopoly','party','friend','table','value','interaction','question','mechanism','small','puzzle', 'rate', 'amount'
         , 'single', 'resource', 'score', 'care', 'able', 'designer', 'took', 'level','seemed', 'help', 'non', 'given'
        ,'person', 'avoid', 'copy', 'taking','edition', 'rulebook', 'color','quickly', 'change', 'sold','complete'
        ,'top','sort', 'event', 'mind', 'http', 'couple', 'ship', 'com', 'unit', 'concept', 'due', 'aspect','battle','often',
         'victory','short', 'special', 'huge','early', 'attack','option','que', 'unplayable', 'winning', 'review', 'run','low',
         'cannot', 'unless', 'designed','comment', 'pure', 'la','name', 'answer','except', 'ability', 'hate', 'absolutely',
     'abstract', 'die', 'adult', 'figure', 'actual', 'age', 'area', 'came'}


# In[26]:


# to clear text from jargon words for better accuraccy

from nltk.corpus import stopwords
def clear_text(text):

    stop = set(stopwords.words('english'))
    
    stop.update(d)
    
    clean_tokens = [tok for tok in text if len(tok.lower())>1 and (tok.lower() not in stop)]

    pos_list = clean_tokens
    
    return pos_list


# In[27]:


#Basic Data Cleaning fucntion

c = 0

def text_preprocess(given_review):
    review = given_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    stop.update(d)
    review = [lemmatizer.lemmatize(w) for w in review if not w in stop]
    global c
    c+=1
    print(c)
    return (' '.join(review))
 


# Now Let us create a new column in the existing reviews_ds['cleaned_text'] dataframe which will contain all cleaned text after applying txt processing

# In[28]:


reviews_ds['cleaned_text'] = reviews_ds['comment'].apply(text_preprocess)


# # Loading the cleaned Dataframe

# In[31]:



reviews_clean = reviews_ds.copy()
reviews_clean = reviews_clean.dropna(axis=0, subset=['cleaned_text'])


# To create word list (positive / negative), I assembled top 100 words using collection.counter list of words for each rating

# In[33]:


"""
EDA Before Data Cleaning 

"""

import numpy as np
from nltk.util import ngrams
from collections import Counter


print('Average word length of phrases in corpus is:',np.mean(reviews_ds['comment'].apply(lambda x: len(x.split()))))


def MostCommonPostEDA(star, no):
    text = ' '.join(reviews_ds.loc[reviews_ds.rating == star, 'comment'].values)
    text_unigrams = [i for i in ngrams(text.split(), 1)]
    text_bigrams = [i for i in ngrams(text.split(), 2)]
    text_trigrams = [i for i in ngrams(text.split(), 3)]
    print("The most common words in rating",star,"\n", Counter(text_unigrams).most_common(no))
    print("The most common bigrams in rating",star, "\n",Counter(text_bigrams).most_common(no))
    print("The most common trigrams in rating",star, "\n",Counter(text_trigrams).most_common(no))
    return Counter(text_unigrams).most_common(no),Counter(text_bigrams).most_common(no),Counter(text_trigrams).most_common(no)


# In[34]:


l1= reviews_ds[reviews_ds["rating"]==1.0]
l1=l1.cleaned_text
a = " ".join(l1).split()
list1= clear_text(a)
str_1=set(Counter(list1).most_common(100))
star_1=[]
for e in str_1:
    star_1.append(e)


# In[35]:


l2= reviews_ds[reviews_ds["rating"]==2.0]
l2=l2.cleaned_text
a = " ".join(l2).split()
list2= clear_text(a)
str_2=sorted(set(Counter(list2).most_common(100)))
star_2=[]
for e in str_2:
    star_2.append(e)


# In[36]:


l3= reviews_ds[reviews_ds["rating"]==3.0]
l3=l3.cleaned_text
a = " ".join(l3).split()
list3= clear_text(a)
str_3=sorted(set(Counter(list3).most_common(100)))
star_3=[]
for e in str_3:
    star_3.append(e)


# In[37]:


l4= reviews_ds[reviews_ds["rating"]==4.0]
l4=l4.cleaned_text
a = " ".join(l4).split()
list4= clear_text(a)
str_4=sorted(set(Counter(list4).most_common(100)))
star_4=[]
for e in str_4:
    star_4.append(e)


# In[38]:


l5= reviews_ds[reviews_ds["rating"]==5.0]
l5=l5.cleaned_text
a = " ".join(l5).split()
list5= clear_text(a)
str_5=sorted(set(Counter(list5).most_common(100)))
star_5=[]
for e in str_5:
    star_5.append(e)


# In[39]:


l6= reviews_ds[reviews_ds["rating"]==6.0]
l6=l6.cleaned_text
a = " ".join(l6).split()
list6= clear_text(a)
str_6=sorted(set(Counter(list6).most_common(100)))
star_6=[]
for e in str_6:
    star_6.append(e)


# In[40]:


l7= reviews_ds[reviews_ds["rating"]==7.0]
l7=l7.cleaned_text
a = " ".join(l7).split()
list7= clear_text(a)
str_7=sorted(set(Counter(list7).most_common(100)))
star_7=[]
for e in str_7:
    star_7.append(e)


# In[41]:


l8= reviews_ds[reviews_ds["rating"]==8.0]
l8=l8.cleaned_text
a = " ".join(l8).split()
list8= clear_text(a)
str_8=sorted(set(Counter(list2).most_common(100)))
star_8=[]
for e in str_8:
    star_8.append(e)


# In[42]:


l9= reviews_ds[reviews_ds["rating"]==9.0]
l9=l9.cleaned_text
a = " ".join(l9).split()
list9= clear_text(a)
str_9=sorted(set(Counter(list9).most_common(100)))
star_9=[]
for e in str_9:
    star_9.append(e)


# In[43]:


l10= reviews_ds[reviews_ds["rating"]==10.0]
l10=l10.cleaned_text
a = " ".join(l10).split()
list10= clear_text(a)
str_10=sorted(set(Counter(list10).most_common(100)))
star_10=[]
for e in str_10:
    star_10.append(e)


# In[44]:


neg_words=[]
for i in range(len(star_1)):
    neg_words.append(star_1[i][0])
    neg_words.append(star_2[i][0])
    neg_words.append(star_3[i][0])
    neg_words.append(star_4[i][0])
    neg_words.append(star_5[i][0])


# # Unique list of negative words with user rating less than or equal to 5 stars

# In[45]:


# convert the set to the list 
unique_neg_list = [] 
      
# traverse for all elements 
for x in neg_words: 
# check if exists in unique_list or not 
    if x not in unique_neg_list: 
        unique_neg_list.append(x) 

    
print(unique_neg_list)


# In[46]:


pos_words=[]
for i in range(len(star_1)):
    pos_words.append(star_10[i][0])
    pos_words.append(star_9[i][0])
    pos_words.append(star_8[i][0])
    pos_words.append(star_7[i][0])
    pos_words.append(star_6[i][0])


# # Unique list of positive words with user rating greater than 5 stars

# In[47]:


# convert the set to the list 
unique_pos_list = [] 
      
# traverse for all elements 
for x in pos_words: 
# check if exists in unique_list or not 
    if x not in unique_pos_list: 
        unique_pos_list.append(x) 

    
print(unique_pos_list)


# In[48]:


'''
Citation: https://stackoverflow.com/questions/13925251/python-bar-plot-from-list-of-tuples

'''

def plot_grams(title,ylab,lis):
  # sort in-place from highest to lowest
  lis.sort(key=lambda x: x[1], reverse=True) 

  # save the names and their respective scores separately
  # reverse the tuples to go from most frequent to least frequent 
  grams = list(zip(*lis))[0]
  count = list(zip(*lis))[1]
  x_pos = np.arange(len(grams)) 

  # calculate slope and intercept for the linear trend line
  slope, intercept = np.polyfit(x_pos, count, 1)
  trendline = intercept + (slope * x_pos)
  plt.plot(x_pos, trendline, color='red', linestyle='--')    
  plt.bar(x_pos, count,align='center')
  plt.xticks(x_pos, grams, rotation=70) 
  plt.ylabel(ylab)
  plt.title(title)
  plt.show()


all_x,all_y,all_z = [],[],[]
stars = ["Star 1","Star 2","Star 3","Star 4"," Star 5"," Star 6"," Star 7"," Star 8"," Star 9"," Star 10"]
#All the common words, unigrams and bigrams in all the sentiments
for i in [1,2,3,4,5]:
    x,y,z = MostCommonPostEDA(i,3)
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)

for x,i in zip(all_x,stars):
  plot_grams(i,"Most Common Words",x)

for y,i in zip(all_y,stars):
  plot_grams(i,"Bi-grams",y)

for z,i in zip(all_z,stars):
  plot_grams(i,"Tri-grams",z)


# In[49]:


'''
Post Cleaning EDA

'''


print('Average word length of phrases in corpus is:',np.mean(reviews_clean['cleaned_text'].apply(lambda x: len(x.split()))))


def MostCommonPostEDA(star, no):
    text = ' '.join(reviews_clean.loc[reviews_clean.rating == star, 'cleaned_text'].values)
    text_unigrams = [i for i in ngrams(text.split(), 1)]
    text_bigrams = [i for i in ngrams(text.split(), 2)]
    text_trigrams = [i for i in ngrams(text.split(), 3)]
    print("The most common words in star",star,"\n", Counter(text_unigrams).most_common(no))
    print("The most common bigrams in star",star, "\n",Counter(text_bigrams).most_common(no))
    print("The most common trigrams in star",star, "\n",Counter(text_trigrams).most_common(no))
    return Counter(text_unigrams).most_common(no),Counter(text_bigrams).most_common(no),Counter(text_trigrams).most_common(no)




all_x,all_y,all_z = [],[],[]
stars = ["Cleaned Star 1","Cleaned Star 2","Cleaned Star 3","Cleaned Star 4","Cleaned Star 5","Cleaned Star 6","Cleaned Star 7","Cleaned Star 8","Cleaned Star 9","Cleaned Star 10"]
#All the common words, unigrams and bigrams in all the sentiments
for i in [1,2,3,4,5,6,7,8,9,10]:
    x,y,z = MostCommonPostEDA(i,3)
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)

for x,i in zip(all_x,stars):
  plot_grams(i,"Most Common Words",x)

for y,i in zip(all_y,stars):
  plot_grams(i,"Bi-grams",y)

for z,i in zip(all_z,stars):
  plot_grams(i,"Tri-grams",z)



# # Data Visualization
#  - representing a visual graph of most common words using wordcloud

# In[50]:


all_words = ' '.join([text for text in reviews_clean['cleaned_text']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')


# In[51]:


positive_words = ' '.join(unique_pos_list)
negative_words = ' '.join(unique_neg_list)


# # Most Positive Words that defines Rating > 5

# In[52]:


pos_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(pos_wordcloud, interpolation="bilinear")
plt.axis('off')


# # Most Negative Words  that defines Rating <= 5

# In[53]:


neg_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(neg_wordcloud, interpolation="bilinear")
plt.axis('off')


# # Conventional Approach
#  - By comparison, the pretrained network's dense layers are replaced with a traditional classifier. The convolutionary base's production is fed directly to the classificators. The traditional classifier is then focused on the features extracted to arrive at a definitive result

# In[54]:


#Splitting into train and test sets for the downsampled data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reviews_ds['cleaned_text'],reviews_ds['rating'], test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[55]:


y_test


# # Baseline Models 
#  - A list of transforms and a final estimator are implemented in sequence. The pipeline's intermediate steps must be 'transforms,' that is, fit and transform methods must be introduced. The final estimator only has to be prepared to enforce.

# In[56]:



from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cv = CountVectorizer(max_features = 50)
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.fit_transform(X_test).toarray()

#Naive Bayes 
from sklearn.naive_bayes import MultinomialNB
classifier_mulnb = MultinomialNB()
classifier_mulnb.fit(X_train_bow, y_train)
y_test_pred_nulnb = classifier_mulnb.predict(X_test_bow)
mulnb_score = accuracy_score(y_test,y_test_pred_nulnb)
print("Multinomial Naive Bayes score", mulnb_score)


# In[57]:


#Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train_bow,  y_train)
y_test_predicted_logreg = log_reg.predict(X_test_bow)
score_test_logreg = accuracy_score(y_test,y_test_predicted_logreg)
print("Logististic Regression score", score_test_logreg)


# In[58]:


#Random Forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_bow,  y_train)
y_test_predicted_rf = rf.predict(X_test_bow)
score_test_rf = metrics.accuracy_score(y_test, y_test_predicted_rf)
print("Random Forest score",score_test_rf)


# In[59]:


#Linear SVC
from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(X_train_bow,  y_train)
y_test_predicted_svm = clf_svm.predict(X_test_bow)
score_test_svm = metrics.accuracy_score(y_test, y_test_predicted_svm)
print("Linear SVM score",score_test_svm)


# In[60]:


#Baseline plot
[mulnb_score,score_test_rf,score_test_logreg,score_test_svm]


# # Pipelines for Hyperparameter tunning
#  - Pipeline offers a simple and intuitive way to organize our ML flows, which are characterized by a consistent sequence of core tasks including pre-processing, selection of features, standardization / normalization and ranking.
#  - Pipeline automates several instances of the fit / transform process by successively calling each estimator to fit, applying transform to input and passing the TfidfVectorizer. 
# 
# In this Expirement the models are tested for different number of features and choose the best fit hyperparameter to test dataset

# In[61]:


#LogisticRegressionclassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

#Contribution 2: to help ML models to fit over different features 
num_fea = [10000,20000,50000,60000,80000,100000,200000,500000]
train_score_log_l = []
test_score_log_l = []
cnf_logreg = []


for i in num_fea:

  log_reg = LogisticRegression(C=1, penalty='l1', solver='liblinear')
  
  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer=None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_log_reg = Pipeline([
      ('tfidf', tfidf),
      ('logreg', log_reg)    ])



  pipeline_log_reg.fit(X_train, y_train)
  train_score_log = pipeline_log_reg.score(X_train, y_train)
  test_score_log = pipeline_log_reg.score(X_test, y_test)
  y_pred_pipeline_log_reg = pipeline_log_reg.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_log_reg,y_test)
  print(classification_report(y_pred_pipeline_log_reg,y_test))
  cnf_logreg.append(cnf)
  train_score_log_l.append(train_score_log)
  test_score_log_l.append(test_score_log)
  
  print(i,"Train score",train_score_log)
  print(i,"Test Score ",test_score_log)
  


# In[62]:


#MultinomialNBClassifier

train_score_mulnb_l = []
test_score_mulnb_l = []
cnf_mulnb = []

for i in num_fea:

  classifier_mulnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer= None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_mulnb = Pipeline([
      ('tfidf', tfidf),
      ('classifier_mulnb', classifier_mulnb)  ])



  pipeline_mulnb.fit(X_train, y_train)
  train_score_mulnb = pipeline_mulnb.score(X_train, y_train)
  test_score_mulnb = pipeline_mulnb.score(X_test, y_test)
  y_pred_pipeline_mulnb = pipeline_mulnb.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_mulnb,y_test)
  print(classification_report(y_pred_pipeline_mulnb,y_test))
  cnf_mulnb.append(cnf)
  train_score_mulnb_l.append(train_score_mulnb)
  test_score_mulnb_l.append(test_score_mulnb)
  print(i,"Train score",train_score_mulnb)
  print(i,"Test Score ",test_score_mulnb)


# In[63]:


#RandomForestClassifier
train_score_rf_l = []
test_score_rf_l = []
cnf_rf = []

for i in num_fea:

  rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 1000)

  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer= None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_rf = Pipeline([
      ('tfidf', tfidf),
      ('classifier_rf', rf)  ])

  pipeline_rf.fit(X_train, y_train)
  train_score_rf = pipeline_rf.score(X_train, y_train)
  test_score_rf = pipeline_rf.score(X_test, y_test)
  y_pred_pipeline_rf = pipeline_rf.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_rf,y_test)
  print(classification_report(y_pred_pipeline_rf,y_test))
  cnf_rf.append(cnf)
  train_score_rf_l.append(train_score_rf)
  test_score_rf_l.append(test_score_rf)
  print(i,"Train score",train_score_rf)
  print(i,"Test Score ",test_score_rf)
  print(i,"Confusion Matrix: \n",cnf_rf)
  


# In[64]:


#LinearSVC

train_score_svc_l = []
test_score_svc_l = []
cnf_svc = []


for i in num_fea:

  svc = LinearSVC(
      C=1.0,
      class_weight='balanced',
      dual=False,
      fit_intercept=True,
      intercept_scaling=1,
      loss='squared_hinge',
      max_iter=2000,
      multi_class='ovr',
      penalty='l2',
      random_state=0,
      tol=1e-05, 
      verbose=0
  )

  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer=None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_svc = Pipeline([
      ('tfidf', tfidf),
      ('svc', svc)    ])



  pipeline_svc.fit(X_train, y_train)
  train_score_svc = pipeline_svc.score(X_train, y_train)
  test_score_svc = pipeline_svc.score(X_test, y_test)
  y_pred_pipeline_svc = pipeline_svc.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_svc,y_test)
  print(classification_report(y_pred_pipeline_svc,y_test))
  cnf_svc.append(cnf)
  train_score_svc_l.append(train_score_svc)
  test_score_svc_l.append(test_score_svc)
  print(i,"Train score",train_score_svc)
  print(i,"Test Score ",test_score_svc)
  


# # Triune Pipelining
#  - Construction of the so-called Triune Pipeline, whose component pipelines can convert raw text into what is no more than the three key building blocks of NLP tasks combined with varied no. of features. 
#  - This feature selection technique works exceptionally well when we deal with enormous dataset that involves redundant training on a dataset to acheieve desirable output
#   - In my case, I trained the cleaned dataset for varied no. of feature pruning
#   
# Each pipeline of components includes a transformer that will output a major type / representation feature in NLP. I also showed that, particularly in conjunction with RandomizedSearchCV, we can combine distinct feature sets resulting from different pipelines to achieve greater results. Another good takeaway I think is the value of combining ML-driven and rule-based methods to boost model performance.

# In[65]:


#Best accuracy comparision
#Values Taken from the above models and rounded to 2 decimals to plot
baseline_scores = [('Logistic Regression',11.2),('Naive Bayes',12.05),('Random Forests',12.19),('Linear SVC',12.9)]
best_scores_tfidf = [('Logistic Regression',33.76),('Naive Bayes',68.86),('Random Forests',18),('Linear SVC',83)]

plot_grams("Baseline (BOW) Comparision","Accuracy",baseline_scores)
plot_grams("TFIDF Pipelines Comparision","Accuracy",best_scores_tfidf)


# In[66]:


import matplotlib.pyplot as plt 



fig, ax = plt.subplots(1, 1 ,figsize=(7,7))
plt.plot(num_fea, train_score_log_l, label='Logistic regression') 
plt.plot(num_fea, train_score_mulnb_l,label='Naive Bayes') 
plt.plot(num_fea, train_score_svc_l,label='Linear SVM Classifier') 
plt.plot(num_fea, train_score_rf_l,label='Random Forests') 


plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
 
plt.xlabel('# of features') 

plt.ylabel('Accuracy') 
 
plt.title('Test Accuracy for different model') 
  
plt.show()


# # Linear SVC Model outstands all other classifiers with Accuracy of 83.2%
# Optimal Hyper-parameter: max-feature = 500000, which is greater than MNB and MNB-N-grams and any other classifiers we tested

# In[67]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

"""

Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""



def plot_confusion_matrix(cm,target_names,title= 'Normalized Confusion matrix',
                          cmap=None,
                          normalize=True):
   

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "gray")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "gray")


    plt.tight_layout()
    
    plt.show()


# # Confusion matrix for LinearSVM showing optimum levels of proficiency

# In[68]:


star = ['1', '2','3','4','5','6','7','8','9','10']

for i in range(len(cnf_svc)): 
  
    if i == (len(cnf_svc)-1): 
            plot_confusion_matrix(cnf_svc[i],star)


# # Findings

# After performing unbiased sampling and Optimizing severals features on classifier, the throughput we obtained was much more accurate and similar to what we as a user might rate in a real world. so, below is an example to demonstrate the working of our model

# In[69]:


pred = pipeline_svc.predict(["it was perfect but it can be better"])[0]
print("Rating = "+str(pred))


# # Use of all data to conduct the final assessment & test for overfit

# In[70]:


##############---Creating test set to predict final daset rating ----########################

from sklearn.utils import shuffle
train = shuffle(data)
X = train['comment'].values
y = train['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[71]:


test_score_svc = pipeline_svc.score(X_test, y_test)
y_pred_pipeline_svc = pipeline_svc.predict(X_test)
cnf = confusion_matrix(y_pred_pipeline_svc,y_test)
print(classification_report(y_pred_pipeline_svc,y_test))
print("Final Test Dataset Accuracy: " +str(test_score_svc))


# In[74]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# Take dump of Best Classifier using Pickle
pickle.dump(pipeline_svc, open('model.pkl', 'wb'))


# # Conclusion & Contributions

# As we can discern a noticeable difference, the unsampled(biased) raw data when passed into the classifier which than hampers our models prediction with a fairly low accuracy of ~16% over sampled train set that we’re able to predict with an accuracy of  ~83.2%.
# 
# Challenges faced
# 
#  - Texual Representation: words having same meaning have a similar representation. It is this approach to representing words and documents that can be considered one of the main breakthroughs of deep learning on the task of solving natural language problems.
#  
# Contributions
# 
# - In this experiment we successfully processed BoardGamesGeek-Reviews Dataset and trained our best fit model to generate pridections for rating based on user's review. 
# - My model involves comparision of baseline classifiers over pipelined models with an approach to a more acustomed feature selection technique, purges all inadequate words in the datalist which does not contribute in prediction. Helps algorithm to learn over relavant words
# - Incorporated several modeling features selection technique to enhance the scope of ML classifiers for better throughput. several classifiers were tested on the same dataset rigorously and compared their test accuracies.
# 
# We can see after training models on same dataset with different attributes imbued varied levels of accuracies which showed an overall hugh success in linear svc model with an exceptional accuracy of 83.2% for trained sample dataset. This model can now be used for prediccting ratings with an optimal score that matches to a great extent with an actual human rating in real world 

# # References:
#  - http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#  - https://towardsdatascience.com/5-minute-guide-to-plotting-with-pandas-e8c0f40a1df4
#  - https://towardsdatascience.com/the-triune-pipeline-for-three-major-transformers-in-nlp-18c14e20530
#  - https://stackoverflow.com/questions/45333530/pandas-drop-columns
#  - https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
#  - https://stackoverflow.com/questions/13925251/python-bar-plot-from-list-of-tuples
#  - https://en.wikipedia.org/wiki/Support-vector_machine
#  - https://medium.com/@Mandysidana/machine-learning-types-of-classification-9497bd4f2e14

# In[ ]:





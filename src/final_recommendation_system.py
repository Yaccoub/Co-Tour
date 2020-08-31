#!/usr/bin/env python
# coding: utf-8


# Standard data science libraries
import pandas as pd
import copy
import numpy as np
from pathlib import Path
import dateparser
from datetime import datetime
from sklearn.impute import SimpleImputer
# Visualization
import seaborn as sns
sns.set()
import matplotlib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
#from countrygroups import EUROPEAN_UNION
import re
import os.path 
get_ipython().run_line_magic('matplotlib', 'inline')
import pycountry
#from googletrans import Translator
import glob
import geopy
import ntpath
from sklearn.cluster import KMeans


# In[33]:


import locale
locale.setlocale(locale.LC_ALL, 'en_US')
EU_countries = EUROPEAN_UNION.names
path = "C:/Users/Oumaima/Documents/AMI/data/*.csv"


# In[34]:


def preprocessing(df):
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Trip type: ': ''}, regex=True)
    df['date']= [datetime.strptime(date, '%B %Y')for date in df['date']]
    df = df.sort_values(by='date', ascending=False, inplace=False, ignore_index=True)
    #df['date'] = df['date'].dt.strftime('%Y-%m')
    df = df.set_index('date')
    

    return df


# In[35]:


def clustering_process(df):
    df[['city', 'country', 'extra']] = df['visitor_origin'].str.split(', ', expand=True, n=2)
    df = df.drop(['rating','title','text'], axis=1)
    return df


# In[36]:


def get_season(df):
    summer_pre_covid = df[(df.index >= '2019-06-1') & (df.index <= '2019-08-01')]
    winter_pre_covid = df[(df.index >= '2019-09-1') & (df.index <= '2020-02-01')]
    winter_post_covid = df[(df.index >= '2020-03-01') & (df.index <= '2020-05-01')]
    summer_post_covid = df[(df.index >= '2020-06-01')]
    
    return summer_pre_covid, winter_pre_covid,winter_post_covid, summer_post_covid


# In[37]:


def feature_extraction(df, file_name):
    df = preprocessing(df)
    df = clustering_process(df)
    visitors_by_country = df.groupby('country').count().sort_values('visit', ascending=True)
    type_of_visitors    = df.groupby('visit').count().sort_values('country', ascending=True)
    type_of_visitors    = type_of_visitors.T.drop(index=['city', 'country' , 'extra'])
    visitors_by_city    = df.groupby('city').count().sort_values('visit', ascending=True)
    type_of_visitors.index.rename(file_name)
    return visitors_by_country, type_of_visitors, visitors_by_city


# In[38]:


def eu_countries(visitors_by_country): 
    visitors_by_country["Non EU"] = 0
    for i in range (len(visitors_by_country)):
        if not(visitors_by_country.index[i] in EU_countries):
            visitors_by_country["Non EU"][i] = int(1)
    return visitors_by_country


# In[39]:


def get_visitors(visitors_by_country, visitors_by_city):
    
    visitors_from_munich    = visitors_by_city['visitor_origin']['Munich']
    visitors_outside_munich = visitors_by_country['visitor_origin']['Germany']- visitors_by_city['visitor_origin']['Munich']
    visitors_outside_eu     = visitors_by_country.groupby('Non EU').sum()['visitor_origin'][1]
    visitors_from_eu        = visitors_by_country.groupby('Non EU').sum()['visitor_origin'][0] - visitors_by_country['visitor_origin']['Germany']
    return visitors_from_munich, visitors_outside_munich, visitors_outside_eu, visitors_from_eu
    


# In[40]:


def binary_encoding(df):
    df = preprocessing(df)
    df = clustering_process(df)
    df = df.reset_index()
    df['provenance'] = ''
    for index, row in df.iterrows():        
        if df['city'][index] == 'Munich':
            df['provenance'][index] = 'Munich'
        elif df['country'][index] == 'Germany' and df['city'][index] != 'Munich':
            df['provenance'][index] = 'outside Munich'
        elif df['country'][index] in EU_countries and df['country'][index] != 'Germany' : 
            df['provenance'][index] = 'EU apart from GER'
        else :
            df['provenance'][index] = 'Outisde EU'
    df = pd.get_dummies(df, columns=["provenance" , "visit"])
    df = df.set_index('date')

    return df


# In[41]:


def get_df_and_names(file_path):
    names = list()
    l_df = list()
    for i in range (len(file_path)-1):
        temp = ntpath.basename(file_path[i])
        names.append(temp[:-4])
    for i in range (len(file_path)-1):
        temp_df = pd.read_csv(file_path[i],  header=0, squeeze=True)
        temp_df['place_name'] = names[i]
        l_df.append(temp_df)
    df = pd.concat(l_df)
    return df, names


# In[42]:


def data_processing(file_path):
    df, names = get_df_and_names(file_path)
    df = binary_encoding(df)
    df = df.reset_index()
    df = df.drop(['visitor_origin','city', 'country', 'country', 'extra', 'date'], axis=1)
    return df, names


# In[43]:


def predict_score(kmeans, df, ori, visit):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = df[df.columns[1:]].index.values
    cluster_map['cluster'] = kmeans.labels_
    cluster_map['place_name'] = df['place_name']
    cluster_i = list()
    ind_i = list()
    for i in range (num_clusters):
        cluster_i.append(cluster_map[cluster_map['cluster'] == i])
        ind_i.append((cluster_map.cluster == i))

    for i in range (num_clusters):
        cluster_i[i] = cluster_i[i]['place_name'].value_counts().reset_index()
        cluster_i[i].columns = ['place_name', 'count']
        cluster_i[i]["count"] = cluster_i[i]["count"] /cluster_i[i]["count"].sum()
        
    for i in range(num_clusters):
        cluster_i[i] = cluster_i[i].rename(columns = {'place_name': 'place_name', 'count': 'score'})

    
    user_eingaben = np.zeros(9)
    
    for i in range (1,len(df.columns)):
        if ori == df.columns[i]:
            user_eingaben[i-1] = 1
        if visit == df.columns[i]:
            user_eingaben[i-1] = 1
    
    return cluster_i[int(kmeans.predict(user_eingaben.reshape(1, -1)))]
    



file_path = glob.glob("C:/Users/Oumaima/Documents/AMI/data/*.csv")
df, names = data_processing(file_path)
num_clusters=10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df[df.columns[1:]])
S = predict_score(kmeans, df, 'provenance_outside Munich', 'visit_Traveled solo')
#S['score']=  (S['score']-S['score'].min())/(S['score'].max()-S['score'].min())
print(S)


# In[45]:


def preprocessing(df):
    df=df.groupby(by = ['place','city_district','type_door'], as_index=False).agg({'rating':'mean','all_metric_score':'mean',})
    df['place_score']=df['all_metric_score']
    return df
def score_func(user,df):
    place_score ={}
    for index, row in df.iterrows():
        place_score[index]=0
        if df['city_district'][index] == user['accomodation']:
            place_score[index] = place_score[index]+10;
        if df['type_door'][index] == user['place_pref']:
            place_score[index] = place_score[index]+10;
    return(place_score)
def reshape_df(df):
    df=df.drop(columns=['city_district','type_door','rating','all_metric_score'])
    df["place_score"] = (df["place_score"] /df["place_score"].sum())*10
    df=df.sort_values(by = "place_score",ascending=False)
    return df


# In[46]:


def get_metrics(df_metrics):
    for index,row in df_metrics.iterrows():
        if '2020-07-01' == df_metrics['DATE'][index]:
            new_listing = df_metrics.loc[index].T
            all_metric_score = df_metrics.loc[index]
    all_metric_score=all_metric_score.replace({0.0: 100000})
    return (all_metric_score)


# In[47]:


def extract_places_features(rec_dataset,metrics):
    rec_dataset=rec_dataset.drop(columns = ['city_district_metric','#_of_visits'])
    places_features = rec_dataset.groupby(by = ['place','city_district','type_door']).agg({'rating':'mean','all_metric_score':'mean'})
    places_features.reset_index(inplace=True, drop=False)
    for index, row in places_features.iterrows():
        for index2, value2 in metrics.items():
            if places_features.loc[index]['place'] == index2 :
                places_features['all_metric_score'][index] = metrics.get(key = index2) 
    places_features['all_metric_score'] =(places_features['all_metric_score']-places_features['all_metric_score'].min())/(places_features['all_metric_score'].max()-places_features['all_metric_score'].min())
    return (places_features)


# In[48]:


def place_type(df):
    outdoors_places = ['Allianz Arena', 'English Garden','Olympiapark', 'Viktualienmarkt','Marienplatz']
    indoors_places = ['Alte Pinakothek','BMW Museum', 'Nymphenburg Palace','Deutsches Museum','Munich Residenz','New_Town_Hall',"St.Peter's Church"]
    return df


# In[49]:


df_metrics = pd.read_csv('C:/Users/Oumaima/Documents/AMI/dataset_predicted.csv')
all_metric_score = get_metrics(df_metrics)
     

rec_dataset = pd.read_csv('C:/Users/Oumaima/Documents/AMI/rec_dataset.csv')
places_features= extract_places_features(rec_dataset,all_metric_score)


# In[50]:



df = preprocessing(places_features)
user = {'origin': 'Berlin', 'accomodation': 'Maxvorstadt', 'visit_type': 'alone', 'place_pref': 'indoors','date':'07.2020'}
for index, row in df.iterrows():
    df['place_score'][index]= score_func(user,df)[index]*10+ df['rating'][index] + df['all_metric_score'][index]*500
dataframe = reshape_df(df)


# In[51]:


def merge_dfs(df1,df2):
    df1.rename(columns={'score':'place_score'}, inplace=True)
    df2.rename(columns={'place':'place_name'}, inplace=True)
    for index,row in df1.iterrows():
        if df1['place_name'][index]=='Deutsches_Museum':
            df1['place_name'][index]='Deutsches Museum'
        if df1['place_name'][index]=='Alte_Pinakothek':
            df1['place_name'][index]='Alte Pinakothek'
        if df1['place_name'][index]=="St.Peter's Church":
            df1['place_name'][index]='St.Peters Church'
    for index1,row1 in df1.iterrows():
        for index2,row2 in df2.iterrows():
           # print(df1['place_name'][index1],df2['place_name'][index2])
            if df2['place_name'][index2]==df1['place_name'][index1]:
                df2['place_score'][index2]=(df1['place_score'][index1]+df2['place_score'][index2])/2
                df1=df1.drop([index1])
                
    return(df1,df2)


# In[52]:


dataframe1, dataframe2 = merge_dfs(S,dataframe)
df=pd.concat([dataframe1,dataframe2]).drop_duplicates(keep=False)



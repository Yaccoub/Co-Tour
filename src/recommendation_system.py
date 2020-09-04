import pandas as pd
import numpy as np

from geopy.geocoders import Nominatim
from datetime import datetime

import seaborn as sns
sns.set()

from countrygroups import EUROPEAN_UNION

import glob
import ntpath
from sklearn.cluster import KMeans
import locale


def preprocessing(df):
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Trip type: ': ''}, regex=True)
    df['date'] = [datetime.strptime(date, '%B %Y') for date in df['date']]
    df = df.sort_values(by='date', ascending=False, inplace=False)
    # df['date'] = df['date'].dt.strftime('%Y-%m')
    df = df.set_index('date')

    return df



def clustering_process(df):
    df[['city', 'country', 'extra']] = df['visitor_origin'].str.split(', ', expand=True, n=2)
    df = df.drop(['rating','title','text'], axis=1)
    return df




def binary_encoding(df):
    EU_countries = EUROPEAN_UNION.names
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
            df['provenance'][index] = 'Outside EU'
    df = pd.get_dummies(df, columns=["provenance" , "visit"])
    df = df.set_index('date')

    return df


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


def data_processing(file_path):
    df, names = get_df_and_names(file_path)
    df = binary_encoding(df)
    df = df.reset_index()
    df = df.drop(['visitor_origin','city', 'country', 'country', 'extra', 'date'], axis=1)
    return df, names


def predict_score(kmeans, df, ori, visit):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = df[df.columns[1:]].index.values
    cluster_map['cluster'] = kmeans.labels_
    cluster_map['place_name'] = df['place_name']
    cluster_i = list()
    ind_i = list()
    for i in range(num_clusters):
        cluster_i.append(cluster_map[cluster_map['cluster'] == i])
        ind_i.append((cluster_map.cluster == i))

    for i in range(num_clusters):
        cluster_i[i] = cluster_i[i]['place_name'].value_counts().reset_index()
        cluster_i[i].columns = ['place_name', 'count']
        cluster_i[i]["count"] = cluster_i[i]["count"] / cluster_i[i]["count"].sum()

    for i in range(num_clusters):
        cluster_i[i] = cluster_i[i].rename(columns={'place_name': 'place_name', 'count': 'score'})

    user_eingaben = np.zeros(9)

    for i in range(1, len(df.columns)):
        if ori == df.columns[i]:
            user_eingaben[i - 1] = 1
        if visit == df.columns[i]:
            user_eingaben[i - 1] = 1

    return cluster_i[int(kmeans.predict(user_eingaben.reshape(1, -1)))]


def update_score(data, user):
    for index, row in data.iterrows():
        for type_door in outdoors_places:
            if data['place_name'][index] == type_door:
                data['place_type'][index] = 'outdoors'
        for type_door in indoors_places:
            if data['place_name'][index] == type_door:
                data['place_type'][index] = 'indoors'
    place_score = {}
    for index, row in data.iterrows():
        place_score[index] = data['score'][index]
        user_preference = user['place_pref']
        if data['place_type'][index] == user_preference:
            place_score[index] *= 2

    return place_score


def preprocessing2(df):
    df=df.groupby(by = ['place','city_district','type_door'], as_index=False).agg({'rating':'mean','all_metric_score':'mean',})
    df['place_score']=df['all_metric_score']
    return df



def score_func(user,df):
    place_score ={}
    for index, row in df.iterrows():
        place_score[index]=0
        if df['city_district'][index] == user['accomodation']:
            place_score[index] = place_score[index]+20;
        if df['type_door'][index] == user['place_pref']:
            place_score[index] = place_score[index]+50;
    return(place_score)



def reshape_df(df):
    df=df.drop(columns=['Unnamed: 0','metric','city_district','type_door','all_metric_score'])
    df["place_score"] = (df["place_score"] -df["place_score"].min())/(df["place_score"].max()-df["place_score"].min())
    df=df.sort_values(by = "place_score",ascending=False)
    return df


def get_metrics(df_metrics,user):

    for index,row in df_metrics.iterrows():
        if user['date'][:7] == df_metrics['DATE'][index][:7]:
            new_listing = df_metrics.loc[index].T
            all_metric_score = df_metrics.loc[index]
    #all_metric_score=all_metric_score.replace({0.0: 100000})
    return (all_metric_score)


def extract_places_features(rec_dataset,metrics):
   # rec_dataset=rec_dataset.drop(columns = ['city_district_metric','#_of_visits'])
   # places_features = rec_dataset.groupby(by = ['place','city_district','type_door']).agg({'rating':'mean','all_metric_score':'mean'})
    #places_features.reset_index(inplace=True, drop=False)
    for index, row in rec_dataset.iterrows():
        for index2, value2 in metrics.items():
            if rec_dataset.loc[index]['place'] == index2 :
                rec_dataset['all_metric_score'][index] = metrics.get(key = index2)
    rec_dataset['all_metric_score'] =(rec_dataset['all_metric_score']-rec_dataset['all_metric_score'].min())/(rec_dataset['all_metric_score'].max()-rec_dataset['all_metric_score'].min())
    return (rec_dataset)



outdoors_places = ['Allianz Arena', 'English Garden', 'Eisbach', 'Marienplatz', 'Olympiaturm',
                       'Olympiastadion', 'Olympiapark', 'Tierpark Hellabrunn', 'Viktualienmarkt']

indoors_places = ['Alte Pinakothek', 'Asamkirche Munich', 'Bayerisches Nationalmuseum', 'Bayerisches Staatsoper',
                      'BMW Museum', 'Deutsches Museum', 'Kleine Olympiahalle', 'Lenbachhaus', 'Museum Mensch und Natur',
                      'Muenchner Stadtmuseum', 'Muenchner Kammerspiele', 'Munich Residenz', 'Muenchner Philharmoniker',
                      'Museum Brandhorst', 'Nationaltheater', 'Neue Pinakothek', 'Neues Rathaus Munich',
                      'Nymphenburg Palace',
                      'Olympiahalle', 'Olympia-Eissportzentrum',
                      'Prinzregententheater', 'Pinakothek der Moderne',
                      'Schack galerie', 'St-Peter Munich', 'Staatstheater am Gaertnerplatz']

def merge_dfs(df1, df2):
    df1.rename(columns={'score': 'place_score'}, inplace=True)
    df2.rename(columns={'place': 'place_name'}, inplace=True)
    df1 = df1.sort_values(by='place_score', ascending=False)
    df2 = df2.sort_values(by='place_score', ascending=False)
    #if df2.place_name.isin(df1.place_name):
    #    df2.place_score = (df1.place_score + df2.place_score) / 2
    for index1, row1 in df1.iterrows():
        for index2, row2 in df2.iterrows():
            if row1.place_name == row2.place_name:
                print(df1.place_name[index1],df2.place_name[index2])
                row2.place_score = (row1.place_score + row2.place_score) / 2
                df1 = df1.drop([index1])
            if (row1.place_name == 'Allianz Arena') & (row2.place_name == 'Olympiastadion'):
                row2.place_score = (row1.place_score + row2.place_score) / 2
                df1 = df1.drop([index1])

    return (df1, df2)



def get_user_country(user_country):
    EU_countries = EUROPEAN_UNION.names
    geolocator = Nominatim(user_agent="AMI")
    location = geolocator.geocode(user_country, language="en")
    if 'Munich' in location.address:
        provenance = 'provenance_Munich'
    elif ('Germany' in location.address) and not('Munich' in location.address):
        provenance = 'outside Munich'
    elif location.address.rsplit(', ')[-1] in EU_countries and location.address.rsplit(', ')[-1] != 'Germany':
        provenance = 'provenance_EU apart from GER'
    else:
        provenance = 'provenance_Outside EU'
    return provenance


locale.setlocale(locale.LC_ALL, 'en_US')
path = "../data/Tripadvisor_datasets/*.csv"
visit_type = 'visit_Traveled with family'
user_country = 'France'
place_pref = 'indoors'
date_of_visit = '2020-08-01'
provenance = get_user_country(user_country)
user = {'origin': provenance, 'accomodation': 'Maxvorstadt', 'visit_type': visit_type, 'place_pref': place_pref,'date':date_of_visit}


file_path = glob.glob("../data/Tripadvisor_datasets/*.csv")
df, names = data_processing(file_path)
df.to_csv('../data/Recommendation data/user_data.csv',index = False)
num_clusters=10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df[df.columns[1:]])
S = predict_score(kmeans, df, user['origin'], user['visit_type'])
new_S=S
new_S['place_type']=new_S['place_name']
score_list_S = update_score(new_S,user)
for index, row in new_S.iterrows():
    new_S['score'][index] = score_list_S[index]
S=new_S.drop(columns=['place_type'])
#print(new_S)

S['score']= (S['score']-S['score'].min())/(S['score'].max()-S['score'].min())


df_metrics = pd.read_csv('../data/Forecast Data/dataset_predicted.csv')
all_metric_score = get_metrics(df_metrics,user)

rec_dataset = pd.read_csv('../data/Recommendation data/rec_dataset.csv')
rec_dataset['all_metric_score'] = 0
rec_dataset['place_score'] = 0
places_features = extract_places_features(rec_dataset, all_metric_score)

df = places_features
score_list = score_func(user,df)
for index, row in df.iterrows():
    df['place_score'][index] = (score_list[index]*10)+ (df['metric'][index] *0.00005 )+ (df['all_metric_score'][index]*0.001)
dataframe = reshape_df(df)

dataframe1, dataframe2 = merge_dfs(S,dataframe)
df1=pd.concat([dataframe1,dataframe2]).drop_duplicates(keep=False)
df1=df1.sort_values(by ='place_score',ascending=False)
df1 = df1.reset_index(drop=True)
print(S)
print(dataframe)
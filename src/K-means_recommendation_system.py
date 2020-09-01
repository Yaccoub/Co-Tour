import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns

sns.set()
from countrygroups import EUROPEAN_UNION
import glob
import ntpath
from sklearn.cluster import KMeans
import locale

locale.setlocale(locale.LC_ALL, 'en_US')
EU_countries = EUROPEAN_UNION.names
path = "../data/Tripadvisor_datasets/*.csv"


def preprocessing(df):
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Trip type: ': ''}, regex=True)
    df['date'] = [datetime.strptime(date, '%B %Y') for date in df['date']]
    df = df.sort_values(by='date', ascending=False, inplace=False, ignore_index=True)
    # df['date'] = df['date'].dt.strftime('%Y-%m')
    df = df.set_index('date')

    return df


def clustering_process(df):
    df[['city', 'country']] = df['visitor_origin'].str.split(', ', expand=True, n=1)
    df = df.drop(['rating', 'title', 'text'], axis=1)
    return df


def eu_countries(visitors_by_country):
    visitors_by_country["Non EU"] = 0
    for i in range(len(visitors_by_country)):
        if not (visitors_by_country.index[i] in EU_countries):
            visitors_by_country["Non EU"][i] = int(1)
    return visitors_by_country


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
        elif df['country'][index] in EU_countries and df['country'][index] != 'Germany':
            df['provenance'][index] = 'EU apart from GER'
        else:
            df['provenance'][index] = 'Outisde EU'
    df = pd.get_dummies(df, columns=["provenance", "visit"])
    df = df.set_index('date')

    return df


def get_df_and_names(file_path):
    names = list()
    l_df = list()
    my_dict = {}
    for i in range(len(file_path)):
        temp = ntpath.basename(file_path[i])
        names.append(temp[:-4])
    for i in range(len(file_path)):
        temp_df = pd.read_csv(file_path[i], header=0, squeeze=True)
        temp_df['place_name'] = names[i]
        l_df.append(temp_df)
        my_dict[names[i]] = temp_df

    df = pd.concat(l_df)
    return df, names, my_dict


def data_processing(file_path):
    df, names, my_dict = get_df_and_names(file_path)
    df = binary_encoding(df)
    df = df.reset_index()
    df = df.drop(['visitor_origin', 'city', 'country', 'country', 'date'], axis=1)
    return df, names, my_dict


def predict_score(kmeans, df, ori, visit, my_dict):
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
        cluster_i[i]["count"]

    for i in range(num_clusters):
        cluster_i[i] = cluster_i[i].rename(columns={'place_name': 'place_name', 'count': 'score'})

    user_eingaben = np.zeros(9)

    for i in range(1, len(df.columns)):
        if ori == df.columns[i]:
            user_eingaben[i - 1] = 1
        if visit == df.columns[i]:
            user_eingaben[i - 1] = 1
    res_sco = cluster_i[int(kmeans.predict(user_eingaben.reshape(1, -1)))]

    list__ = list()
    for i in range(len(res_sco)):
        list__.append(res_sco['score'][i] / len(my_dict[res_sco['place_name'][i]]))
    res_sco['score'] = list__

    return res_sco


"""Accepted strings are 
- ori :
'provenance_EU apart from GER'
'provenance_Munich'
'provenance_Outisde EU'
'provenance_outside Munich'
- visit :
'visit_Traveled as a couple'
'visit_Traveled on business'
'visit_Traveled solo'
'visit_Traveled with family'
'visit_Traveled with friends'"""

# examples:
provenance = 'provenance_outside Munich'
visit = 'visit_Traveled as a couple'

file_path = glob.glob(path)
df, names, my_dict = data_processing(file_path)
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df[df.columns[1:]])
S = predict_score(kmeans, df, provenance, visit, my_dict)
print(S)

import pandas as pd
from datetime import datetime
from countrygroups import EUROPEAN_UNION
import os.path
import glob
import locale
from sklearn.cluster import KMeans
import geopy

locale.setlocale(locale.LC_ALL, 'en_US')
EU_countries = EUROPEAN_UNION.names
path = "C:/Users/yacco/Documents/TUM/Applied Machine Learning/group16/data/Tripadvisor_datasets/*.csv"

def preprocessing(df):
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['date'] = df['date'].replace({'Date of experience: ': ''}, regex=True)
    df['visit'] = df['visit'].replace({'Trip type: ': ''}, regex=True)
    df['date']= [datetime.strptime(date, '%B %Y')for date in df['date']]
    df = df.sort_values(by='date', ascending=False, inplace=False, ignore_index=True)
    #df['date'] = df['date'].dt.strftime('%Y-%m')
    df = df.set_index('date')

    return df

def clustering_process(df):
    df[['city', 'country']] = df['visitor_origin'].str.split(', ', expand=True, n=1)
    df = df.drop(['rating','title','text'], axis=1)
    return df


def get_season(df):
    summer_pre_covid = df[(df.index >= '2019-06-1') & (df.index <= '2019-08-01')]
    winter_pre_covid = df[(df.index >= '2019-09-1') & (df.index <= '2020-02-01')]
    winter_post_covid = df[(df.index >= '2020-03-01') & (df.index <= '2020-05-01')]
    summer_post_covid = df[(df.index >= '2020-06-01')]

    return summer_pre_covid, winter_pre_covid, winter_post_covid, summer_post_covid


def feature_extraction(df, file_name):
    df = preprocessing(df)
    df = clustering_process(df)
    visitors_by_country = df.groupby('country').count().sort_values('visit', ascending=True)
    type_of_visitors    = df.groupby('visit').count().sort_values('country', ascending=True)
    type_of_visitors    = type_of_visitors.T.drop(index=['city', 'country'])
    visitors_by_city    = df.groupby('city').count().sort_values('visit', ascending=True)

    type_of_visitors.index.rename(file_name)

    return visitors_by_country, type_of_visitors, visitors_by_city

def eu_countries(visitors_by_country):
    visitors_by_country["Non EU"] = 0
    for i in range (len(visitors_by_country)):
        if not(visitors_by_country.index[i] in EU_countries):
            visitors_by_country["Non EU"][i] = int(1)
    return visitors_by_country


def get_visitors(visitors_by_country, visitors_by_city):
    visitors_from_munich = visitors_by_city['visitor_origin']['Munich']
    visitors_outside_munich = visitors_by_country['visitor_origin']['Germany'] - visitors_by_city['visitor_origin'][
        'Munich']
    visitors_outside_eu = visitors_by_country.groupby('Non EU').sum()['visitor_origin'][1]
    visitors_from_eu = visitors_by_country.groupby('Non EU').sum()['visitor_origin'][0] - \
                       visitors_by_country['visitor_origin']['Germany']
    return visitors_from_munich, visitors_outside_munich, visitors_outside_eu, visitors_from_eu

def kmeans(data):
    kmeans = KMeans(n_clusters=3)
    data = data.fillna(0)
    data['Cluster'] = kmeans.fit_predict(data)
    data.to_csv('clusters.csv')
    return data


def get_file(path):
    file_names = []
    data = pd.DataFrame()
    names = glob.glob(path)
    for i in range(len(names)):
        df = pd.read_csv(names[i], header=0, squeeze=True)
        file_name = os.path.basename(names[i])
        file_name = file_name.split('.')[0]
        file_names.append(file_name)
        visitors_by_country, type_of_visitors, visitors_by_city = feature_extraction(df, file_name)
        visitors_by_country = eu_countries(visitors_by_country)
        visitors_from_munich, visitors_outside_munich, visitors_outside_eu, visitors_from_eu = get_visitors(
            visitors_by_country, visitors_by_city)
        type_of_visitors['visitors_from_munich'] = visitors_from_munich
        type_of_visitors['visitors_outside_munich'] = visitors_outside_munich
        type_of_visitors['visitors_outside_eu'] = visitors_outside_eu
        type_of_visitors['visitors_from_eu'] = visitors_from_eu
        type_of_visitors['attraction_name'] = file_name
        data = data.append(type_of_visitors)

        print("Attraction %s is being processed..." % (str(file_name)))
    data.reset_index()
    data.set_index('attraction_name', inplace=True)
    data[['Traveled on business', 'Traveled solo', 'Traveled with friends', 'Traveled as a couple',
          'Traveled with family']] = data[
        ['Traveled on business', 'Traveled solo', 'Traveled with friends', 'Traveled as a couple',
         'Traveled with family']].div(data[['Traveled on business', 'Traveled solo', 'Traveled with friends',
                                            'Traveled as a couple', 'Traveled with family']].sum(axis=1), axis=0)
    data[['visitors_from_munich', 'visitors_outside_munich', 'visitors_outside_eu', 'visitors_from_eu']] = data[
        ['visitors_from_munich', 'visitors_outside_munich', 'visitors_outside_eu', 'visitors_from_eu']].div(
        data[['visitors_from_munich', 'visitors_outside_munich', 'visitors_outside_eu', 'visitors_from_eu']].sum(
            axis=1), axis=0)
    data.to_csv('k_means_data.csv')

    return data


data = get_file(path)
clusters = kmeans(data)

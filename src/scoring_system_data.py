import glob
from pathlib import Path

import pandas as pd

import dateparser
from datetime import timedelta
from datetime import datetime

def special_characters_col(data):
    """Function to get rid of the special characters in tripadvisor dataset"""
    ret = []
    for index, row in df.iterrows():
        ret.append(
            data.iloc[index].place.replace('Ä', 'AE').replace('Ö', 'OE').replace('Ü', 'UE').replace('ä', 'ae').replace(
                'ö', 'oe').replace('ü', 'ue').replace('Ã¼', 'ue').replace('Ã¤', 'ae'))
    ret = pd.DataFrame(ret, columns=['place'])
    data = data.assign(place=ret['place'], city_district=ret['place'])
    return data
outdoors_places = ['Allianz Arena', 'English Garden', 'Eisbach', 'Marienplatz', 'Olympiaturm',
                   'Olympiastadion', 'Olympiapark', 'Tierpark Hellabrunn', 'Viktualienmarkt' ]

indoors_places = ['Alte Pinakothek', 'Asamkirche Munich', 'Bayerisches Nationalmuseum', 'Bayerisches Staatsoper',
                  'BMW Museum', 'Deutsches Museum', 'Kleine Olympiahalle', 'Lenbachhaus', 'Museum Mensch und Natur',
                  'Muenchner Stadtmuseum', 'Muenchner Kammerspiele', 'Munich Residenz', 'Muenchner Philharmoniker',
                  'Museum Brandhorst', 'Nationaltheater', 'Neue Pinakothek', 'Neues Rathaus Munich',
                  'Nymphenburg Palace',
                  'Olympiahalle', 'Olympia-Eissportzentrum',
                  'Prinzregententheater', 'Pinakothek der Moderne',
                  'Schack galerie', 'St-Peter Munich', 'Staatstheater am Gaertnerplatz']
dataset = pd.read_csv("../data/Forecast Data/dataset.csv")
# Tripadvisor setup
path = "../data/Tripadvisor_datasets/*.csv"
dataframe = pd.DataFrame()

# Read and reformat tripadvisor data
for fname in glob.glob(path):
    #if Path(fname).stem in dataset.columns:
    print(Path(fname).stem)
    x = pd.read_csv(fname, low_memory=False)
    x = x.dropna(subset=['date'])
    x['date'] = [date.replace('Date of experience: ', '') for date in x['date']]
    x['date'] = [dateparser.parse(date).strftime('%d/%m/%Y') for date in x['date']]
    x['date'] = [datetime.strptime(date, '%d/%m/%Y') for date in x['date']]
    x['place'] = Path(fname).stem
    x['visit'].fillna('', inplace=True)
    x['visit'] = [visit_type.replace('Trip Type: ', '') for visit_type in x['visit']]
    x = x[['date', 'place', 'rating', 'visit']]
    dataframe = pd.concat([dataframe, x], axis=0)

# Grouping the tripadvisor datasets
df = dataframe.groupby(['date', 'place'], as_index=False)[['rating']].mean()
df2 = dataframe.groupby(['date', 'place'])[['date']].count()
df['#_of_visits'] = df2['date'].values
# df['city_district'] = df['place']
df['date'] = df['date'] + timedelta(days=1)
df = special_characters_col(df)
# Special character treatment

geo_coords = pd.read_csv('../data/geocoordinates/geoattractions.csv', low_memory=False)
geo_coords = geo_coords.set_index('place')
places = df['place'].unique()

# for place in places:
#     df['city_district'] =

#
for index, row in df.iterrows():
    df['city_district'][index] = geo_coords.loc[row['place']]['city_district']
# take the metrics at June 2020 for all places ( from dataset.csv)
all_metric_score = dataset.tail(1)
# df is dataframe taken from data_forecast code
# add a new column with the city_distict_metric

df['city_district_metric'] = df['city_district']
df2 = df
df2['all_metric_score'] = ''
for index, row in df2.iterrows():
    #if df2['city_district'][index] != '':
    #    df2['city_district_metric'][index] = neigh_metric_score.iloc[0][df2['city_district'][index]]
    try:
        df2['all_metric_score'][index] = all_metric_score.iloc[0][df2['place'][index]]
    except:
        df2['all_metric_score'][index] = 0
# add a new column with the type_door : indoors or outdoors
df2['type_door'] = df2['place']
for index, row in df2.iterrows():
    for type_door in outdoors_places:
        if df2['place'][index] == type_door:
            df2['type_door'][index] = 'outdoors'
    for type_door in indoors_places:
        if df2['place'][index] == type_door:
            df2['type_door'][index] = 'indoors'
# store the dataset into a csv file for further use
rec_dataset = df2
# Read munich visitor data
state = pd.read_csv('../data/munich_visitors/munich_visitors.csv', engine='python')
state = state.drop(['Ausland (Tourismus)', 'Inland (Tourismus)', 'Kinos', 'Muenchner Philharmoniker',
                    'Schauburg - Theater fuer junges Publikum'], axis=1)
state['Year'] = state['DATE'][4:]
state = state.groupby(['Year']).sum()
state=state.T
state = state.reset_index(drop=False)
state['Total']=''
state.loc[:,'Total'] = state.sum(axis=1)
state = state[['index', 'Total']]
state.rename(columns={'index': 'place','Total':'metric'}, inplace=True)
state['city_district']=''
for index, row in state.iterrows():
    state['city_district'][index] = geo_coords.loc[row['place']]['city_district']
# take the metrics at June 2020 for all places ( from dataset.csv)
state['type_door'] = state['place']
for index, row in state.iterrows():
    for type_door in outdoors_places:
        if state['place'][index] == type_door:
            state['type_door'][index] = 'outdoors'
    for type_door in indoors_places:
        if state['place'][index] == type_door:
            state['type_door'][index] = 'indoors'
state.to_csv('../data/Recommendation data/rec_dataset.csv')
#rec_dataset.to_csv('../data/Recommendation data/rec_dataset_new.csv')

import glob
from pathlib import Path
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import dateparser
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import preprocessing

state = pd.read_csv('../data/munich_visitors/Features.csv', low_memory=False)
state = state.set_index('DATE')
for i in range(len(state)):
    state.iloc[i]= state.iloc[i]/state.iloc[i]['insgesamt']
path = "../Tripadvisor_web_scraper/data/*.csv"
dataframe = pd.DataFrame()
for fname in glob.glob(path):
    x = pd.read_csv(fname, low_memory=False)
    x = x.dropna(subset=['date'])
    x['date'] = [date.replace('Erlebnisdatum: ', '') for date in x['date']]
    x['date'] = [dateparser.parse(date).strftime('%Y.%m')  for date in x['date']]
    x['place'] = Path(fname).stem
    x['visit'].fillna('', inplace=True)
    x['visit'] = [visit_type.replace('Reiseart: ', '') for visit_type in x['visit']]
    x = x[['date', 'place', 'rating', 'visit']]
    dataframe = pd.concat([dataframe, x], axis=0)

df = dataframe.groupby(['date', 'place'], as_index=False)[['rating']].mean()
df2 = dataframe.groupby(['date', 'place'])[['date']].count()
df['#_of_visits'] = df2['date'].values
df['city_district'] =df['place']

city_district = ['Altstadt-Lehel', 'Ludwigsvorstadt-Isarvorstadt', 'Maxvorstadt', 'Schwabing-West'
    , 'Au-Haidhausen'
    , 'Sendling'
    , 'Sendling-Westpark'
    , 'Schwanthalerhöhe'
    , 'Neuhausen-Nymphenburg'
    , 'München-Moosach'
    , 'Milbertshofen-Am Hart'
    , 'Schwabing-Freimann'
    , 'Bogenhausen'
    , 'Berg am Laim'
    , 'Trudering-Riem'
    , 'Ramersdorf-Perlach'
    , 'Obergiesing'
    , 'Untergiesing-Harlaching'
    , 'Thalkirchen-Obersendling-Forstenried-Fürstenried-Solln'
    , 'Hadern'
    , 'Pasing-Obermenzing'
    , 'Aubing-Lochhausen-Langwied'
    , 'Allach-Untermenzing'
    , 'Feldmoching-Hasenbergl'
    , 'Laim']
geolocator = Nominatim(user_agent='salut', timeout=3)

places = df['place'].unique()
y = {}
for place in places:
    try:
        print(place)
        location = geolocator.geocode(place, addressdetails=True, country_codes='de')
        location2 = location.address
        for district in city_district:
            if district in location2:
                y[place] = district
    except:
        y[place] = ''

for index, row in df.iterrows():
        df['city_district'][index] = y[row['place']]

nparray = np.array(df)
ret = []

for place in places:
    name = place
    tmp = pd.DataFrame(df[df.place == name])
    tmp = tmp.set_index('date')
    del tmp['place']
    mat = pd.DataFrame(index= tmp.index)
    mat[place] = tmp.apply(lambda x: [np.array(x)], axis=1).apply(lambda x: x[0])
    ret.append(mat)


# Concat the feature table

df_clean = pd.concat(ret, axis=1, sort=True)
for place in df_clean.columns:
    if place in state.columns:
        df_clean[place][:][0] = state[place]

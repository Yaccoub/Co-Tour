import glob
from pathlib import Path
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import dateparser
import ast
from datetime import datetime
from datetime import timedelta

def special_characters_col(data):
    """Function to get rid of the special characters in tripadvisor dataset"""
    ret = []
    for index, row in df.iterrows():
        ret.append(data.iloc[index].place.replace('Ä','AE').replace('Ö','OE').replace('Ü','UE').replace('ä','ae').replace('ö','oe').replace('ü','ue').replace('Ã¼','ue').replace('Ã¤','ae'))
    ret = pd.DataFrame(ret, columns=['place'])
    data = data.assign(place=ret['place'], city_district=ret['place'])
    return data

# Read munich visitor data
state = pd.read_csv('../data/munich_visitors/munich_visitors.csv', engine='python')
state['DATE'] = [datetime.strptime(date, '%d/%b/%Y') for date in state['DATE']]
state = state.set_index('DATE')

# Normalizing munich visitor data
#TODO: The error is located here
for i in range(len(state)):
    state.iloc[i] = state.iloc[i] / state.iloc[i].sum()

# Read airbnb data
listings = pd.read_csv('../data/Airbnb_data/listings.csv', low_memory=False)
listings['Datum']= [datetime.strptime(date, '%d/%b/%Y')for date in listings['Datum']]
listings = listings.set_index('Datum')

# Normalizing airbnb data
for i in range(len(listings)):
    for district in listings.columns:
        listings.iloc[i][district] = ast.literal_eval(listings.iloc[i][district])
        listings.iloc[i][district]= listings.iloc[i][district][0]* 10 + listings.iloc[i][district][1]
    #TODO: Write the normalization somewhre else
    listings.iloc[i] = listings.iloc[i] / listings.iloc[i].sum()

# Tripadvisor setup
path = "../Tripadvisor_web_scraper/data/*.csv"
dataframe = pd.DataFrame()

# Read and reformate tripadvisor data
for fname in glob.glob(path):
    x = pd.read_csv(fname, low_memory=False)
    x = x.dropna(subset=['date'])
    x['date'] = [date.replace('Erlebnisdatum: ', '') for date in x['date']]
    x['date'] = [dateparser.parse(date).strftime('%d/%m/%Y') for date in x['date']]
    x['date'] = [datetime.strptime(date, '%d/%m/%Y') for date in x['date']]
    x['place'] = Path(fname).stem
    x['visit'].fillna('', inplace=True)
    x['visit'] = [visit_type.replace('Reiseart: ', '') for visit_type in x['visit']]
    x = x[['date', 'place', 'rating', 'visit']]
    dataframe = pd.concat([dataframe, x], axis=0)

# Grouping the tripadvisor datasets
df = dataframe.groupby(['date', 'place'], as_index=False)[['rating']].mean()
df2 = dataframe.groupby(['date', 'place'])[['date']].count()
df['#_of_visits'] = df2['date'].values
df['city_district'] = df['place']

# Special character treatment
df = special_characters_col(df)

city_district = ['Altstadt-Lehel', 'Ludwigsvorstadt-Isarvorstadt', 'Maxvorstadt', 'Schwabing-West'
    , 'Au-Haidhausen', 'Sendling', 'Sendling-Westpark', 'Schwanthalerhöhe'
    , 'Neuhausen-Nymphenburg', 'Muenchen-Moosach', 'Milbertshofen-Am Hart', 'Schwabing-Freimann'
    , 'Bogenhausen', 'Berg am Laim', 'Trudering-Riem', 'Ramersdorf-Perlach', 'Obergiesing'
    , 'Untergiesing-Harlaching', 'Thalkirchen-Obersendling-Forstenried-Fürstenried-Solln', 'Hadern'
    , 'Pasing-Obermenzing', 'Aubing-Lochhausen-Langwied', 'Allach-Untermenzing', 'Feldmoching-Hasenbergl'
    , 'Laim']
places = df['place'].unique()
y = {}
geolocator = Nominatim(user_agent='salut', timeout=3)

for place in places:
    try:
        print(place)
        location = geolocator.geocode(place, addressdetails=True, country_codes='de')
        location2 = location.address
        for district in city_district:
            if district in location2:
                y[place] = district
    except:
        #TODO: No entry in: Bayerisches Staatsorchester, Muenchner Philharmoniker, Staedtische Galerie im Lenbachhaus
        y[place] = ''

for index, row in df.iterrows():
    df['city_district'][index] = y[row['place']]

ret = []
for place in places:
    name = place
    tmp = pd.DataFrame(df[df.place == name])
    tmp = tmp.set_index('date')
    del tmp['place']
    tmp['popularity_metric'] = tmp['rating'] * tmp['#_of_visits']

    mat = pd.DataFrame(index=tmp.index)
    mat[place] = tmp['popularity_metric']
    ret.append(mat)

# Concat the feature table
df_clean = pd.concat(ret, axis=1, sort=True)
#TODO: Write the normalization somewhre else
for i in range(len(df_clean)):
    df_clean.iloc[i] = df_clean.iloc[i] / df_clean.iloc[i].sum()

for place in places:
    if y[place] != '':
        arr = np.array([df_clean[place][listings.index],state[place][listings.index],listings[y[place]]])
        arr = pd.DataFrame(arr).replace(float('nan'), np.nan)
        state[place][listings.index] = arr.mean()

dataset = pd.DataFrame(state)
dataset.div(dataset.sum(axis=1), axis=0)

# Read COVID-19 data
Covid_19 = pd.read_csv('../data/covid_19_data/rki/COVID_19_Cases_SK_Muenchen.csv', low_memory=False)
Covid_19['Refdatum'] = [datetime.strptime(date, '%Y-%m-%d') for date in Covid_19['Refdatum']]
Covid_19 = Covid_19.set_index('Refdatum')
Covid_19 = Covid_19.resample('1M').sum()
Covid_19.index = Covid_19.index + timedelta(days=1)

# Create the complete dataset
dataset = pd.concat([dataset, Covid_19], axis=1)
dataset = dataset.reset_index()
dataset = dataset.rename(columns={"index": "DATE"})
dataset['AnzahlFall'] = dataset['AnzahlFall'].fillna(0)
#TODO: Replace NaN values

# Drop the last rows as this data is not complete
dataset = dataset.fillna(0)[:-3]
dataset.to_csv('../data/Forecast Data/dataset.csv', index=False)


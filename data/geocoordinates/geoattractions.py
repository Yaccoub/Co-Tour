import pandas as pd
from geopy.geocoders import Nominatim
import glob
from pathlib import Path
from datetime import datetime
from datetime import timedelta

geolocator = Nominatim(user_agent="AMI", timeout=3)

#setup
path = "../Tripadvisor_datasets/*.csv"
Attractions = pd.DataFrame()
list__ = list()

# Read and reformate tripadvisor data
for fname in glob.glob(path):
    df = pd.DataFrame()
    list__.append(Path(fname).stem)

state = pd.read_csv('../munich_visitors/munich_visitors.csv', engine='python')
state['DATE'] = [datetime.strptime(date, '%d/%m/%Y') for date in state['DATE']]
state = state.set_index('DATE')
state = state.drop(['Ausland (Tourismus)', 'Inland (Tourismus)', 'Kinos', 'Muenchner Philharmoniker', 'Schauburg - Theater fuer junges Publikum'], axis = 1)
for x in state.columns:
    if x not in list__:
        list__.append(x)
Attractions['place'] = list__
Attractions['latitude'] = Attractions.place.apply(lambda x: geolocator.geocode(x).latitude)
Attractions['longitude'] = Attractions.place.apply(lambda x: geolocator.geocode(x).longitude)

city_district = ['Altstadt-Lehel', 'Ludwigsvorstadt-Isarvorstadt', 'Maxvorstadt', 'Schwabing-West'
    , 'Au-Haidhausen', 'Sendling', 'Sendling-Westpark', 'Schwanthalerhöhe'
    , 'Neuhausen-Nymphenburg', 'Muenchen-Moosach', 'Milbertshofen-Am Hart', 'Schwabing-Freimann'
    , 'Bogenhausen', 'Berg am Laim', 'Trudering-Riem', 'Ramersdorf-Perlach', 'Obergiesing'
    , 'Untergiesing-Harlaching', 'Thalkirchen-Obersendling-Forstenried-Fürstenried-Solln', 'Hadern'
    , 'Pasing-Obermenzing', 'Aubing-Lochhausen-Langwied', 'Allach-Untermenzing', 'Feldmoching-Hasenbergl'
    , 'Laim']
places = Attractions['place'].unique()
y = {}
for place in places:
    print(place)
    location = geolocator.geocode(place, addressdetails=True, country_codes='de')
    try:
        location2 = location.address
        for district in city_district:
            if district in location2:
                y[place] = district
    except:
        #TODO: No entry in: Bayerisches Staatsorchester, Muenchner Philharmoniker, Staedtische Galerie im Lenbachhaus
        y[place] = ''

y['Eisbach'] = "Schwabing-Freimann"
Attractions['city_district'] = Attractions['place']
for index, row in Attractions.iterrows():
    Attractions.loc[index,'city_district'] = y[row['place']]

Attractions.to_csv('./geoattractions.csv', index=False)
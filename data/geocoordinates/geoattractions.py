import pandas as pd
from geopy.geocoders import Nominatim
import glob
from pathlib import Path
from datetime import datetime
from datetime import timedelta

geolocator = Nominatim(user_agent="AMI", timeout=3)

# setup
path = "../Tripadvisor_datasets/*.csv"
Attractions = pd.DataFrame()
Attractions_tripadvisor = pd.DataFrame()
Attractions_state = pd.DataFrame()
list__ = list()

# Read and reformate tripadvisor data
for fname in glob.glob(path):
    list__.append(Path(fname).stem)

Attractions_tripadvisor['place'] = list__
Attractions_tripadvisor['latitude'] = Attractions_tripadvisor.place.apply(lambda x: geolocator.geocode(x).latitude)
Attractions_tripadvisor['longitude'] = Attractions_tripadvisor.place.apply(lambda x: geolocator.geocode(x).longitude)

state = pd.read_csv('../munich_visitors/munich_visitors.csv', engine='python')
state['DATE'] = [datetime.strptime(date, '%d/%m/%Y') for date in state['DATE']]
state = state.set_index('DATE')
state = state.drop(['Ausland (Tourismus)', 'Inland (Tourismus)', 'Kinos', 'Muenchner Philharmoniker',
                    'Schauburg - Theater fuer junges Publikum'], axis=1)
Attractions_state['place'] = state.columns
Attractions_state['latitude'] = Attractions_state.place.apply(lambda x: geolocator.geocode(x).latitude)
Attractions_state['longitude'] = Attractions_state.place.apply(lambda x: geolocator.geocode(x).longitude)

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
districtList__ = {}
addressList__ = {}
for place in places:
    print(place)
    location = geolocator.geocode(place, addressdetails=True, country_codes='de')
    try:
        location2 = location.address
        for district in city_district:
            if district in location2:
                districtList__[place] = district
        addressList__[place] = location2
    except:
        # TODO: No entry in: Bayerisches Staatsorchester, Muenchner Philharmoniker, Staedtische Galerie im Lenbachhaus
        districtList__[place] = ''
        addressList__[place] = ''

districtList__['Eisbach'] = "Schwabing-Freimann"
Attractions['city_district'] = Attractions['place']
Attractions_tripadvisor['city_district'] = Attractions_tripadvisor['place']
Attractions_state['city_district'] = Attractions_state['place']

Attractions['address'] = Attractions['place']
Attractions_tripadvisor['address'] = Attractions_tripadvisor['place']
Attractions_state['address'] = Attractions_state['place']
for index, row in Attractions.iterrows():
    Attractions.loc[index, 'city_district'] = districtList__[row['place']]
    Attractions.loc[index, 'address'] = addressList__[row['place']]

for index, row in Attractions_tripadvisor.iterrows():
    Attractions_tripadvisor.loc[index, 'city_district'] = districtList__[row['place']]
    Attractions_tripadvisor.loc[index, 'address'] = addressList__[row['place']]

for index, row in Attractions_state.iterrows():
    Attractions_state.loc[index, 'city_district'] = districtList__[row['place']]
    Attractions_state.loc[index, 'address'] = addressList__[row['place']]

Attractions = Attractions.sort_values('place')
Attractions_tripadvisor = Attractions_tripadvisor.sort_values('place')
Attractions_state = Attractions_state.sort_values('place')

Attractions.to_csv('./geoattractions.csv', index=False)
Attractions_tripadvisor.to_csv('./TripAdvisor_geoattractions.csv', index=False)
Attractions_state.to_csv('./State_geoattractions.csv', index=False)

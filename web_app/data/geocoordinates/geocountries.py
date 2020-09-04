from geopy import Nominatim
import pandas as pd

geolocator = Nominatim(user_agent="AMI")


def Geolocate(df):
    s = df.shape[0]
    for i in range(s):
        location = geolocator.geocode(df['country_name'][i])
        if location is None:
            continue
        df.loc[i, ['Latitude', 'Longitude']] = (location.latitude, location.longitude)


countries = pd.read_csv('./country.csv')
countries = countries.rename(columns={'value': 'country_name'})
countries.insert(2, 'Latitude', 0.0)
countries.insert(3, 'Longitude', 0.0)
Geolocate(countries)
countries = countries.drop('id', axis=1)
countries = countries[countries.Latitude != 0.0]
countries.to_csv('./Geocoordinates')

import pandas as pd
import numpy as np
import math
from geopy.geocoders import Nominatim
import folium
import pycountry
from datetime import datetime

geolocator = Nominatim(user_agent="AMI")


def get_season(df):
    df = df[['date', 'text', 'visitor_origin']]
    df.date = df.date.replace({'Date of experience: ': ''}, regex=True)
    # df.date = df.date.replace('Date of experience: ', '')
    df.date = [datetime.strptime(date, '%B %Y') for date in df['date']]
    summer_pre_covid = df[(df.date >= '2019-06-1') & (df.date <= '2019-08-01')]
    winter_pre_covid = df[(df.date >= '2019-09-1') & (df.date <= '2020-02-01')]
    winter_covid = df[(df.date >= '2020-03-01') & (df.date <= '2020-05-01')]
    summer_covid = df[(df.date >= '2020-06-01')]

    return summer_pre_covid, winter_pre_covid, winter_covid, summer_covid


def origin(df):
    # extract relevant columns of dataframe
    df = df[['date', 'text', 'visitor_origin']]
    df = df.rename(columns={'text': 'review_number'})
    # drop all rows without an origin
    df = df.dropna(subset=['visitor_origin'])
    ## a correct origin has the format "City, Region, Country"
    #  by using rpartition you create 3 strings that split at the last comma
    # if only the country is given, it will return two empty strings and the last one with the country
    df[['city', 'region', 'country']] = df.visitor_origin.apply(
        lambda x: pd.Series(str(x).rpartition(",")))
    # now that the last word has been extracted, other columns can be deleted
    df = df.drop(["visitor_origin", 'city', 'region'], axis=1)
    # delete leading and trailing spaces in preparation for grouping
    df.country = df.country.apply(lambda x: x.strip())
    # group countries together and count the number of visitors per country
    df = df.groupby(by='country', as_index=False).agg({'review_number': 'count'}).sort_values(by='country',
                                                                                              ascending=False)
    # check that each country given is valid using the pycountry library
    df.insert(2, 'valid_country', np.nan)
    df.valid_country = df.country.apply(lambda x: countrycheck(x))
    # drop the rows with unvalid countries
    df = df.dropna(subset=['valid_country'])
    df = df.drop(['valid_country'], axis=1)
    # compute the percentage of visitors from that country to the attraction
    df['flux density'] = df['review_number'] * 100 / sum(df['review_number'])
    df = df.drop(['review_number'], axis=1)
    # specific case problem where geopy assigns the geocoordinates of Georgia (state in the USA) and not of Georgia the country
    # so I specifically ask for the coordinates of the capital of Georgia
    df.loc[df['country'] == 'Georgia', 'country'] = 'Tbilisi, Georgia'
    # assign latitude and longitude values to each country
    df['latitude'] = df.country.apply(lambda x: geolocator.geocode(x).latitude)
    df['longitude'] = df.country.apply(lambda x: geolocator.geocode(x).longitude)
    # rename Georgia correctly for visualisation
    df.loc[df['country'] == 'Tbilisi, Georgia', 'country'] = 'Georgia'
    return df


def markersize(number):
    size = 2 + math.ceil(number)
    return size


def visualise(df):
    # create a map centered on munich
    map1 = folium.Map(
        location=[geolocator.geocode('Munich, Germany').latitude, geolocator.geocode('Munich, Germany').longitude],
        tiles='cartodbpositron',
        zoom_start=4,
        max_zoom=6,
        min_zoom=2,
    )
    # add a marker on each country propotional to the number of visitors to the selected location
    df.apply(lambda row: folium.CircleMarker(radius=markersize(row["flux density"]),
                                             location=[row["latitude"], row["longitude"]], tooltip=str(
            round(row["flux density"], 1)) + '% of visitors originate from ' + str(row["country"])).add_to(map1),
             axis=1)
    return map1


def countrycheck(text):
    # check that the given country is valid
    if pycountry.countries.get(name=text) is not None:
        return True



def main():
    path = "../data/Tripadvisor_datasets/*.csv"

    # Read and reformate tripadvisor data
    for fname in glob.glob(path):
        x = pd.read_csv(fname, low_memory=False)

        spc,wpc,wc,sc = get_season(x)

        processed_spc =origin(spc)
        processed_spc.to_csv('../data/Tripadvisor_datasets/Seasons/{}_summer_pre_covid.csv'.format(fname))

        processed_wpc =origin(wpc)
        processed_wpc.to_csv('../data/Tripadvisor_datasets/Seasons/{}_winter_pre_covid.csv'.format(fname))

        processed_sc=origin(sc)
        processed_sc.to_csv('../data/Tripadvisor_datasets/Seasons/{}_summer_covid.csv'.format(fname))

        processed_wc=origin(wc)
        processed_wc.to_csv('../data/Tripadvisor_datasets/Seasons/{}_winter_covid.csv'.format(fname))

    df= pd.read_csv('../data/Tripadvisor_datasets/Seasons/Marienplatz_winter_covid.csv')
    visualise(df)

if __name__ == '__main__':
    main()

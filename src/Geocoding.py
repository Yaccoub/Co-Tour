import os
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sb
import statistics
import pandas as pd
import numpy as np
import math
import geopandas as gpd
from pandas.plotting import lag_plot
from geopy.geocoders import Nominatim
from flask import Flask
import folium
import pycountry
from googletrans import Translator

geolocator = Nominatim(user_agent="AMI")


def origin(df):
    # extract relevant columns of dataframe
    df = df[['text', 'visitor_origin']]
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
                                             location=[geolocator.geocode(row["country"]).latitude,
                                                       geolocator.geocode(row["country"]).longitude], tooltip=str(
            round(row["flux density"], 1)) + '% of visitors originate from ' + str(row["country"])).add_to(map1),
             axis=1)
    return map1


def countrycheck(text):
    # check that the given country is valid
    if pycountry.countries.get(name=text) is not None:
        return True

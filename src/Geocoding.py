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


def Sequence(df):
    df = df[['date', 'text', 'visitor_origin']]
    df.date = df.date.replace('Date of experience:', '')
    df[['d', 'date']] = df.date.apply(
        lambda x: pd.Series(str(x).split(":")))
    df_new = df.drop(["d"], axis=1)
    df_new[['d', 'f', 'day']] = df_new.date.apply(
        lambda x: pd.Series(str(x).rpartition("-")))
    df_new = df_new.drop(['f', 'day'], axis=1)
    df_new.dropna(axis='index', how='any')
    df_new[df_new.visitor_origin.str.contains(",", na=False)]
    df_new[['city', 'land', 'country']] = df_new.visitor_origin.apply(
        lambda x: pd.Series(str(x).rpartition(",")))
    df_new = df_new.drop(["visitor_origin", 'city', 'land'], axis=1)
    df_new_a = df_new.groupby(by='country', as_index=False).agg({'text': 'count'}).sort_values(by='country',
                                                                                               ascending=False)
    df_new_a.head(100)
    df_new_a['flux density'] = df_new_a['text'] * 100 / sum(df_new_a['text'])
    df_new_a['flux density'] = df_new_a['text'] * 100 / sum(df_new_a['text'])
    df_new_a.insert(3, 'Latitude', 0.0)
    df_new_a.insert(4, 'Longitude', 0.0)
    return df_new_a


def Geolocate(df):
    s = df.shape[0]
    for i in range(s):
        location = geolocator.geocode(df['country'][i])
        if location is None:
            continue
        df.loc[i, ['Latitude', 'Longitude']] = (location.latitude, location.longitude)


def Markersize(number):
    size = 2 + 2 * math.ceil(number)
    return size


def Visualise(df):
    map1 = folium.Map(
        location=[geolocator.geocode('Munich, Germany').latitude, geolocator.geocode('Munich, Germany').longitude],
        tiles='cartodbpositron',
        zoom_start=4,
        max_zoom=6,
        min_zoom=2,
    )
    df.apply(lambda row: folium.CircleMarker(radius=Markersize(row["flux density"]),
                                             location=[row["Latitude"], row["Longitude"]], tooltip=str(
            round(row["flux density"], 1)) + '% of visitors originate from ' + str(row["country"])).add_to(map1),
             axis=1)
    return map1
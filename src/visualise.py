import pandas as pd
import math
from geopy.geocoders import Nominatim
import folium

geolocator = Nominatim(user_agent="AMI")

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

map1


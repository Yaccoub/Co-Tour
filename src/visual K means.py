import pandas as pd
import glob
from geopy.geocoders import Nominatim
import folium

def visualise(df):
    # create a map centered on munich
    map1 = folium.Map(
        location=[48.137154,11.576124],
        tiles='cartodbpositron',
        zoom_start=4,
        max_zoom=20,
        min_zoom=12
    )
    # add a marker on each country propotional to the number of visitors to the selected location
    data.apply(lambda row: folium.CircleMarker(
        location=[row[x], row[y]], color=row["color"],
        fill=True, tooltip='this is the' + str(row["attraction_name"]), radius=4).add_to(map_), axis=1)

    return map1




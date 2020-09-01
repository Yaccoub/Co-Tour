import numpy as np
import pandas as pd
import folium

data = pd.read_csv("../data/K_means_data/clusters.csv")
geocoordinates = pd.read_csv('../data/geocoordinates/geoattractions.csv')
data['latitude']= geocoordinates['latitude']
data['longitude']= geocoordinates['longitude']
x, y = "latitude", "longitude"
color = "Cluster"
lst_elements = sorted(list(data[color].unique()))
lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in
              range(len(lst_elements))]
data["color"] = data[color].apply(lambda x:
                lst_colors[lst_elements.index(x)])
# initialize the map with the starting location
map_ = folium.Map(location=[48.137154,11.576124], tiles="cartodbpositron",
                  zoom_start=13)## add points
data.apply(lambda row: folium.CircleMarker(
    location=[row[x],row[y]], color=row["color"],
    fill=True, tooltip='this is the' + str(row["attraction_name"]) ,radius=8).add_to(map_), axis=1)
map_


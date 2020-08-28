import math
from datetime import datetime

import folium
import pandas as pd
from django.views.generic import TemplateView
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="UX")


def get_geo_data(geolocator):
    dataset = pd.read_csv('../data/Forecast Data/dataset_predicted.csv')
    dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
    dataset = dataset.set_index('DATE')

    geo = pd.DataFrame(index=dataset.columns[:-1])
    geo['Longitude'] = ''
    geo['Latitude'] = ''
    geo['Weights'] = ''
    for place in geo.index:
        print(place)
        geo_info = geolocator.geocode(query=place, timeout=3)
        try:
            geo['Latitude'][place] = geo_info.latitude
            geo['Longitude'][place] = geo_info.longitude
        except:
            geo['Latitude'][place] = ''
            geo['Longitude'][place] = ''
        geo['Weights'] = dataset.loc['2020-02-01']

    geo = geo[geo['Longitude'].astype(bool)]

    geo['Weights'] = geo['Weights'] * 100
    geo.sort_values(by='Weights', ascending=False)

    return geo


class HomeView(TemplateView):
    template_name = 'webapp/home.html'


class TfaView(TemplateView):
    template_name = 'webapp/tourist_flow_analysis.html'


class ThfView(TemplateView):
    template_name = 'webapp/tourist_hotspot_forecast.html'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.geolocator = Nominatim(user_agent="UX")

    def top_ten_place(self, **kwargs):
        geo = get_geo_data(geolocator)
        top_10 = geo.head(10)

        Row_list = []

        # Iterate over each row
        for index, rows in top_10.iterrows():
            my_list = [index]
            Row_list.append(my_list)
        return Row_list

    def get_map(self, **kwargs):
        figure = folium.Figure()
        lat = 48.137154
        lon = 11.576124
        m = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=7,
        )
        m.add_to(figure)
        # geo = self.get_geo_data()
        # geo.apply(lambda row: folium.CircleMarker(radius=(2 + 2 * math.ceil(row["Weights"])),
        #                                           location=[row["Latitude"], row["Longitude"]], tooltip=str(
        #         row["Weights"])).add_to(m), axis=1)
        figure.render()
        return figure

    def get_map2(self, **kwargs):
        figure = folium.Figure()
        lat = 48.137154
        lon = 11.576124
        m = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=7,
        )
        m.add_to(figure)
        figure.render()
        return figure

    def get_context_data(self, **kwargs):
        context = super(ThfView, self).get_context_data(**kwargs)
        figure = self.get_map()
        figure2 = self.get_map2()
        context['map'] = figure
        context['map2'] = figure2
        context['Text'] = self.top_ten_place()
        return context

import math
from datetime import datetime

import folium
import pandas as pd
from django.views.generic import TemplateView
from geopy.geocoders import Nominatim


# Create your views here.


class HomeView(TemplateView):
    template_name = 'webapp/home.html'


class TfaView(TemplateView):
    template_name = 'webapp/tourist_flow_analysis.html'


class ThfView(TemplateView):
    template_name = 'webapp/tourist_hotspot_forecast.html'

    def get_context_data(self, **kwargs):
        figure = folium.Figure()
        lat = 48.137154;
        lon = 11.576124
        m = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=7,
        )
        geolocator = Nominatim(user_agent="UX")
        m.add_to(figure)
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
        geo.apply(lambda row: folium.CircleMarker(radius=(2 + 2 * math.ceil(row["Weights"])),
                                                  location=[row["Latitude"], row["Longitude"]], tooltip=str(
                row["Weights"])).add_to(m), axis=1)

        figure.render()
        return {"map": figure}

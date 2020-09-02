import math
from datetime import datetime
import numpy as np
import folium
import pandas as pd
import simplejson
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.views.generic import TemplateView
from geopy.geocoders import Nominatim
import sys
import glob
import locale

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

sys.path.insert(1, '../src')
from recommendation_system import binary_encoding

def load_state_geo_coords():
    geo_coords = pd.read_csv('../data/geocoordinates/State_geoattractions.csv', low_memory=False)
    geo_coords = geo_coords.set_index('place')

    return geo_coords


def load_tripadvisor_geo_coords():
    geo_coords = pd.read_csv('../data/geocoordinates/TripAdvisor_geoattractions.csv', low_memory=False)
    geo_coords = geo_coords.set_index('place')
    return geo_coords


def load_data(type):
    if type == 'pred':
        dataset = pd.read_csv('../data/Forecast Data/dataset_predicted.csv')

        dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
        dataset = dataset.set_index('DATE')
    elif type == 'hist':
        dataset = pd.read_csv('../data/Forecast Data/dataset.csv')

        dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
        dataset = dataset.set_index('DATE')
    return dataset


def load_countries_list():
    dataset = pd.read_csv('../data/geocoordinates/country.csv')
    return dataset


def get_geo_data_historical(dataset, date):
    colors_list = [
        'darkblue',
        'blue',
        'lightblue',
        'cadetblue',
        'green',
        'lightgreen',
        'orange',
        'lightred',
        'red',
        'darkred',
        'darkblue'
    ]
    geo_coords = load_state_geo_coords()

    geo = pd.DataFrame(index=dataset.columns[:-1])
    geo['Longitude'] = ''
    geo['Latitude'] = ''
    geo['Weights'] = ''
    for place in geo.index:
        try:
            geo['Latitude'][place] = geo_coords['latitude'][place]
            geo['Longitude'][place] = geo_coords['longitude'][place]
        except:
            geo['Latitude'][place] = ''
            geo['Longitude'][place] = ''
        geo['Weights'] = dataset.loc[date]
    geo = geo[geo['Longitude'].astype(bool)]

    geo['Weights'] = geo['Weights'] * 100
    geo.sort_values(by='Weights', ascending=False)
    geo['Place'] = geo.index
    geo["Weights"] = minmax_scale(geo["Weights"])

    temp_colors_list = list()
    for i in range(len(geo)):
        temp_colors_list.append(colors_list[int(geo["Weights"][i] * 10)])
    geo['Color'] = temp_colors_list
    return geo


def get_geo_data_predicted(dataset, date):
    colors_list = [
        'darkblue',
        'blue',
        'lightblue',
        'cadetblue',
        'green',
        'lightgreen',
        'orange',
        'lightred',
        'red',
        'darkred',
        'darkblue'
    ]
    geo_coords = load_state_geo_coords()

    geo = pd.DataFrame(index=dataset.columns[:-1])
    geo['Longitude'] = ''
    geo['Latitude'] = ''
    geo['Weights'] = ''
    for place in geo.index:
        try:
            geo['Latitude'][place] = geo_coords['latitude'][place]
            geo['Longitude'][place] = geo_coords['longitude'][place]
        except:
            geo['Latitude'][place] = ''
            geo['Longitude'][place] = ''
        geo['Weights'] = dataset.loc[date]
    geo = geo[geo['Longitude'].astype(bool)]

    geo['Weights'] = geo['Weights'] * 100

    geo.sort_values(by='Weights', ascending=False)
    geo['Place'] = geo.index
    geo["Weights"] = minmax_scale(geo["Weights"])

    temp_colors_list = list()
    for i in range(len(geo)):
        temp_colors_list.append(colors_list[int(geo["Weights"][i] * 10)])
    geo['Color'] = temp_colors_list
    return geo


def get_flow_data():
    data = pd.read_csv("../data/K_means_data/clusters.csv")
    data = data.set_index('attraction_name')
    data['Place'] = data.index
    data['Longitude'] = ''
    data['Latitude'] = ''

    return data


class HomeView(TemplateView):
    template_name = 'webapp/home.html'


class TfaView(TemplateView):
    template_name = 'webapp/tourist_flow_analysis.html'

    def get_season(self):
        tfa_season_select = self.request.GET.get('tfa_season_select', 'summer_pre_covid')
        print(tfa_season_select)
        return tfa_season_select

    def get_place(self):
        tfa_place_select = self.request.GET.get('tfa_place_select', 'Olympiapark')
        print(tfa_place_select)
        return tfa_place_select

    def get_map(self, df, **geo):
        lst_elements = sorted(list(df['Cluster'].unique()))
        lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in
                      range(len(lst_elements))]
        df['Color'] = df['Cluster'].apply(lambda x:
                                          lst_colors[lst_elements.index(x)])
        figure = folium.Figure()
        lat = 48.137154
        lon = 11.576124
        m = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=12,
        )
        df.apply(lambda row: folium.CircleMarker(radius=8,color=row['Color'],
                                                 location=[row['Latitude'], row['Longitude']], fill=True,
                                                 fill_color= row['Color'], tooltip=str(row["Place"])).add_to(m),
                 axis=1)
        m.add_to(figure)
        figure.render()
        return figure

    def get_map2(self, df, **kwargs):
        figure = folium.Figure()
        lat = 48.137154
        lon = 11.576124
        map1 = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=4,
        )
        # add a marker on each country propotional to the number of visitors to the selected location
        df.apply(lambda row: folium.CircleMarker(radius=2 + math.ceil(row["flux density"]),
                                                 location=[row["latitude"], row["longitude"]], fill=True,
                                                 fill_color='#3186cc', tooltip=str(
                round(row["flux density"], 1)) + '% of visitors originate from ' + str(row["country"])).add_to(map1),
                 axis=1)
        map1.add_to(figure)
        figure.render()
        return figure

    def get_context_data(self, **kwargs):
        context = super(TfaView, self).get_context_data(**kwargs)
        geo_coords = load_tripadvisor_geo_coords()
        geo_flow = get_flow_data()

        for place in geo_flow.index:
            try:
                geo_flow['Latitude'][place] = geo_coords['latitude'][place]
                geo_flow['Longitude'][place] = geo_coords['longitude'][place]
            except:
                geo_flow['Latitude'][place] = ''
                geo_flow['Longitude'][place] = ''
        PlacesList = geo_coords.index
        SeasonList = {'summer_pre_covid': 'Summer 2019', 'winter_pre_covid': 'Winter 2019',
                      'summer_covid': 'Summer 2020', 'winter_covid': 'Winter 2020'}
        season = self.get_season()
        place = self.get_place()

        geo_trajectory = pd.read_csv('../data/Tripadvisor_datasets/Seasons/{}.csv_{}.csv'.format(place, season))
        figure = self.get_map(geo_flow)
        figure2 = self.get_map2(geo_trajectory)
        context['map'] = figure
        context['map2'] = figure2
        context['selected_season'] = season
        context['selected_place'] = place
        context['PlacesList'] = PlacesList
        context['SeasonList'] = SeasonList
        return context


class ThfView(TemplateView):
    template_name = 'webapp/tourist_hotspot_forecast.html'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_date_forecast(self):
        tfh_month_select = self.request.GET.get('tfh_month_select', 'Jul 2020')
        tfh_month_select = datetime.strptime(tfh_month_select, '%b %Y')
        tfh_month_select = tfh_month_select.strftime("%Y-%m-%d")
        print(tfh_month_select)
        return tfh_month_select

    def get_date_hist(self):
        month_picker = self.request.GET.get('month_picker', 'Apr 2020')
        month_picker = datetime.strptime(month_picker, '%b %Y')
        month_picker = month_picker.strftime("%Y-%m-%d")
        print(month_picker)
        return month_picker

    def top_ten_place(self, geo, **kwargs):
        top_10 = geo.head(10)
        Row_list = []
        # Iterate over each row
        for index, rows in top_10.iterrows():
            my_list = [index]
            Row_list.append(my_list)
        return Row_list

    def get_map(self, geo, **kwargs):
        figure = folium.Figure()
        lat = 48.137154
        lon = 11.576124
        m = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=12,
        )
        m.add_to(figure)
        geo.apply(lambda row: folium.Marker(icon=folium.Icon(color=row['Color']),
                                            location=[row["Latitude"], row["Longitude"]], tooltip='<div class="card">' + str(
                row['Place']) + str(row["Weights"]) + '</div>').add_to(m), axis=1)
        figure.render()
        return figure

    def get_map_hist(self, geo, **kwargs):
        figure = folium.Figure()
        lat = 48.137154
        lon = 11.576124
        m = folium.Map(
            location=[lat, lon],
            tiles='cartodbpositron',
            zoom_start=12,
        )
        m.add_to(figure)
        geo.apply(lambda row: folium.Marker(icon=folium.Icon(color=row['Color']),
                                            location=[row["Latitude"], row["Longitude"]], tooltip=str(
                row['Place']) + str(row["Weights"])).add_to(m), axis=1)
        figure.render()
        return figure

    def get_context_data(self, **kwargs):
        context = super(ThfView, self).get_context_data(**kwargs)

        dataset_pred = load_data('pred')
        PredDateList = dataset_pred.index.strftime("%b %Y")
        date_pred = self.get_date_forecast()
        geo = get_geo_data_predicted(dataset_pred, date_pred)
        figure = self.get_map(geo)
        top_10 = self.top_ten_place(geo)

        dataset_hist = load_data('hist')
        HistDateList = dataset_hist.index.strftime("%b %Y")
        date_hist = self.get_date_hist()
        geo = get_geo_data_historical(dataset_hist, date_hist)
        figure2 = self.get_map_hist(geo)

        context['map'] = figure
        context['map2'] = figure2
        context['PredDateList'] = PredDateList
        context['HistDateList'] = HistDateList
        date_pred = datetime.strptime(date_pred, '%Y-%m-%d')
        date_hist = datetime.strptime(date_hist, '%Y-%m-%d')
        context['selected_pred_date'] = date_pred.strftime("%b %Y")
        context['selected_hist_date'] = date_hist.strftime("%b %Y")
        context['Text'] = top_10  # self.top_ten_place()
        return context


class TrsView(TemplateView):
    template_name = 'webapp/tourism_recommendation_system.html'
    """def get_context_data(self, **kwargs):
        locale.setlocale(locale.LC_ALL, 'en_US')
        visit_type = 'visit_Traveled with family'
        user_country = 'France'
        place_pref = 'outdoors'
        date_of_visit = '2020-07-01'
        provenance = recommendation_system.get_user_country(user_country)

        user = {'origin': provenance, 'accomodation': 'Maxvorstadt', 'visit_type': visit_type, 'place_pref': place_pref,
                'date': date_of_visit}

        file_path = glob.glob("../data/Tripadvisor_datasets/*.csv")
        df, names = recommendation_system.data_processing(file_path)

        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df[df.columns[1:]])
        S = recommendation_system.predict_score(kmeans, df, user['origin'], user['visit_type'])
        S['score'] = (S['score'] - S['score'].min()) / (S['score'].max() - S['score'].min())

        df_metrics = pd.read_csv('../data/Forecast Data/dataset_predicted.csv')
        all_metric_score = recommendation_system.get_metrics(df_metrics, user)

        rec_dataset = pd.read_csv('../data/Recommendation data/rec_dataset.csv')
        places_features = recommendation_system.extract_places_features(rec_dataset, all_metric_score)

        df = recommendation_system.preprocessing2(places_features)

        for index, row in df.iterrows():
            df['place_score'][index] = recommendation_system.score_func(user, df)[index] * 10 + df['rating'][index] + df['all_metric_score'][
                index] * 0.001
        dataframe = recommendation_system.reshape_df(df)

        dataframe1, dataframe2 = recommendation_system.merge_dfs(S, dataframe)
        df = pd.concat([dataframe1, dataframe2]).drop_duplicates(keep=False)
        df = df.sort_values(by='place_score', ascending=False)
        df = df.reset_index(drop=True)
        context['result'] = df[df.index[0:3]]

        return context"""



class ContactView(TemplateView):
    template_name = 'webapp/contact.html'

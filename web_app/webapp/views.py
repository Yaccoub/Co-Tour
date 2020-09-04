import math
from datetime import datetime
import numpy as np
import folium
import pandas as pd

from django.views.generic import TemplateView
from geopy.geocoders import Nominatim
from countrygroups import EUROPEAN_UNION
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale


def load_geo_coords():
    geo_coords = pd.read_csv('./data/geocoordinates/geoattractions.csv', low_memory=False)
    geo_coords = geo_coords.set_index('place')

    return geo_coords


def load_state_geo_coords():
    geo_coords = pd.read_csv('./data/geocoordinates/State_geoattractions.csv', low_memory=False)
    geo_coords = geo_coords.set_index('place')

    return geo_coords


def load_tripadvisor_geo_coords():
    geo_coords = pd.read_csv('./data/geocoordinates/TripAdvisor_geoattractions.csv', low_memory=False)
    geo_coords = geo_coords.set_index('place')
    return geo_coords


def load_data(type):
    if type == 'pred':
        dataset = pd.read_csv('./data/Forecast Data/dataset_predicted.csv')

        dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
        dataset = dataset.set_index('DATE')
    elif type == 'hist':
        dataset = pd.read_csv('./data/Forecast Data/dataset.csv')

        dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
        dataset = dataset.set_index('DATE')
    return dataset


def load_countries_list():
    dataset_countries = pd.read_csv('./data/geocoordinates/country.csv')
    dataset_germanCities = pd.read_csv('./data/geocoordinates/germanCities.csv')
    dataset_countries = dataset_countries.set_index('value')
    dataset_germanCities = dataset_germanCities.set_index('city')
    return dataset_countries, dataset_germanCities


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


def predict_score(kmeans, df, ori, visit, num_clusters):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = df[df.columns[1:]].index.values
    cluster_map['cluster'] = kmeans.labels_
    cluster_map['place_name'] = df['place_name']
    cluster_i = list()
    ind_i = list()
    for i in range(num_clusters):
        cluster_i.append(cluster_map[cluster_map['cluster'] == i])
        ind_i.append((cluster_map.cluster == i))

    for i in range(num_clusters):
        cluster_i[i] = cluster_i[i]['place_name'].value_counts().reset_index()
        cluster_i[i].columns = ['place_name', 'count']
        cluster_i[i]["count"] = cluster_i[i]["count"] / cluster_i[i]["count"].sum()

    for i in range(num_clusters):
        cluster_i[i] = cluster_i[i].rename(columns={'place_name': 'place_name', 'count': 'score'})

    user_eingaben = np.zeros(9)

    for i in range(1, len(df.columns)):
        if ori == df.columns[i]:
            user_eingaben[i - 1] = 1
        if visit == df.columns[i]:
            user_eingaben[i - 1] = 1

    return cluster_i[int(kmeans.predict(user_eingaben.reshape(1, -1)))]


def score_func(user, df):
    place_score = {}
    for index, row in df.iterrows():
        place_score[index] = 0
        if df['city_district'][index] == user['accomodation']:
            place_score[index] = place_score[index] + 20;
        if df['type_door'][index] == user['place_pref']:
            place_score[index] = place_score[index] + 50;
    return place_score


def reshape_df(df):
    df=df.drop(columns=['Unnamed: 0','metric','city_district','type_door','all_metric_score'])
    df["place_score"] = minmax_scale(df["place_score"])
    df = df.sort_values(by="place_score", ascending=False)
    return df


def get_metrics(df_metrics, user):
    for index, row in df_metrics.iterrows():
        if user['date'][:7] == df_metrics['DATE'][index][:7]:
            all_metric_score = df_metrics.loc[index]
    return all_metric_score


def extract_places_features(rec_dataset, metrics, user):
    for index, row in rec_dataset.iterrows():
        for index2, value2 in metrics.items():
            if rec_dataset.loc[index]['place'] == index2:
                rec_dataset['all_metric_score'][index] = metrics.get(key=index2)
    rec_dataset['all_metric_score'] = (rec_dataset['all_metric_score'] - rec_dataset['all_metric_score'].min()) / (
                rec_dataset['all_metric_score'].max() - rec_dataset['all_metric_score'].min())
    return rec_dataset


def merge_dfs(df1, df2):
    df1.rename(columns={'score': 'place_score'}, inplace=True)
    df2.rename(columns={'place': 'place_name'}, inplace=True)
    df1 = df1.sort_values(by='place_score', ascending=False)
    df2 = df2.sort_values(by='place_score', ascending=False)
    for index1, row1 in df1.iterrows():
        for index2, row2 in df2.iterrows():
            if row1.place_name == row2.place_name:
                print(df1.place_name[index1], df2.place_name[index2])
                row2.place_score = (row1.place_score + row2.place_score) / 2
                df1 = df1.drop([index1])
            if (row1.place_name == 'Allianz Arena') & (row2.place_name == 'Olympiastadion'):
                row2.place_score = (row1.place_score + row2.place_score) / 2
                df1 = df1.drop([index1])

    return df1, df2


def get_user_country(user_country):
    EU_countries = EUROPEAN_UNION.names
    geolocator = Nominatim(user_agent="AMI")
    location = geolocator.geocode(user_country, language="en")
    if 'Munich' in location.address:
        provenance = 'provenance_Munich'
    elif ('Germany' in location.address) and not ('Munich' in location.address):
        provenance = 'outside Munich'
    elif location.address.rsplit(', ')[-1] in EU_countries and location.address.rsplit(', ')[-1] != 'Germany':
        provenance = 'provenance_EU apart from GER'
    else:
        provenance = 'provenance_Outside EU'
    return provenance


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

    geo = geo.sort_values(by='Weights', ascending=False)
    geo['Place'] = geo.index
    geo["Weights"] = minmax_scale(geo["Weights"])

    temp_colors_list = list()
    for i in range(len(geo)):
        temp_colors_list.append(colors_list[int(geo["Weights"][i] * 10)])
    geo['Color'] = temp_colors_list
    geo = geo.sort_values(by='Weights', ascending=False)
    return geo


def get_flow_data():
    data = pd.read_csv("./data/K_means_data/clusters.csv")
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
        df.apply(lambda row: folium.CircleMarker(radius=8, color=row['Color'],
                                                 location=[row['Latitude'], row['Longitude']], fill=True,
                                                 fill_color=row['Color'], tooltip=str(row["Place"])).add_to(m),
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

        geo_trajectory = pd.read_csv('./data/Tripadvisor_datasets/Seasons/{}_{}.csv'.format(place, season))
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
            my_list = index
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
                                            location=[row["Latitude"], row["Longitude"]],
                                            tooltip='<div class="card">' + str(
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
        context['top_10'] = top_10
        return context


class TrsView(TemplateView):
    template_name = 'webapp/tourism_recommendation_system.html'

    def update_score(self,data, user):

        outdoors_places = ['Allianz Arena', 'English Garden', 'Eisbach', 'Marienplatz', 'Olympiaturm',
                           'Olympiastadion', 'Olympiapark', 'Tierpark Hellabrunn', 'Viktualienmarkt']

        indoors_places = ['Alte Pinakothek', 'Asamkirche Munich', 'Bayerisches Nationalmuseum',
                          'Bayerisches Staatsoper',
                          'BMW Museum', 'Deutsches Museum', 'Kleine Olympiahalle', 'Lenbachhaus',
                          'Museum Mensch und Natur',
                          'Muenchner Stadtmuseum', 'Muenchner Kammerspiele', 'Munich Residenz',
                          'Muenchner Philharmoniker',
                          'Museum Brandhorst', 'Nationaltheater', 'Neue Pinakothek', 'Neues Rathaus Munich',
                          'Nymphenburg Palace',
                          'Olympiahalle', 'Olympia-Eissportzentrum',
                          'Prinzregententheater', 'Pinakothek der Moderne',
                          'Schack galerie', 'St-Peter Munich', 'Staatstheater am Gaertnerplatz']

        for index, row in data.iterrows():
            for type_door in outdoors_places:
                if data['place_name'][index] == type_door:
                    data['place_type'][index] = 'outdoors'
            for type_door in indoors_places:
                if data['place_name'][index] == type_door:
                    data['place_type'][index] = 'indoors'
        place_score = {}
        for index, row in data.iterrows():
            place_score[index] = data['score'][index]
            user_preference = user['place_pref']
            if data['place_type'][index] == user_preference:
                place_score[index] *= 2
        return place_score

    def get_country(self):
        trs_country_select = self.request.GET.get('trs_country_select', 'Tunisia')
        print(trs_country_select)
        return trs_country_select

    def get_gerCity(self):
        trs_city_select = self.request.GET.get('trs_city_select', 'Munich')
        print(trs_city_select)
        return trs_city_select

    def get_visit(self):
        trs_visit_select = self.request.GET.get('trs_visit_select', 'solo')
        trs_visit_select = 'visit_Traveled ' + trs_visit_select
        print(trs_visit_select)
        return trs_visit_select

    def get_accommodation(self):
        trs_accommodation_select = self.request.GET.get('trs_accommodation_select', 'Maxvorstadt')
        print(trs_accommodation_select)
        return trs_accommodation_select

    def get_date_visit(self):
        date_picker = self.request.GET.get('date_picker', '2020-08-10')
        print(date_picker)
        return date_picker

    def get_preference(self):
        trs_preferences_select = self.request.GET.get('trs_preferences_select', 'outdoors')
        print(trs_preferences_select)
        return trs_preferences_select


    def get_context_data(self, **kwargs):
        context = super(TrsView, self).get_context_data(**kwargs)
        # locale.setlocale(locale.LC_ALL, 'en_US')
        country_list, cities_list = load_countries_list()
        CountriesList = country_list.index
        GermanCitiesList = cities_list.index
        DistrictList = ['Altstadt-Lehel', 'Ludwigsvorstadt-Isarvorstadt', 'Maxvorstadt', 'Schwabing-West'
            , 'Au-Haidhausen', 'Sendling', 'Sendling-Westpark', 'Schwanthalerhöhe'
            , 'Neuhausen-Nymphenburg', 'Muenchen-Moosach', 'Milbertshofen-Am Hart', 'Schwabing-Freimann'
            , 'Bogenhausen', 'Berg am Laim', 'Trudering-Riem', 'Ramersdorf-Perlach', 'Obergiesing'
            , 'Untergiesing-Harlaching', 'Thalkirchen-Obersendling-Forstenried-Fürstenried-Solln', 'Hadern'
            , 'Pasing-Obermenzing', 'Aubing-Lochhausen-Langwied', 'Allach-Untermenzing', 'Feldmoching-Hasenbergl'
            , 'Laim']
        country = self.get_country()

        if country == 'Germany':
            gerCity = self.get_gerCity()
            context['selected_city'] = gerCity
            user_location = gerCity
        else:
            user_location = country

        visit_type = self.get_visit()
        place_pref = self.get_preference()
        date_of_visit = self.get_date_visit()
        provenance = get_user_country(user_location)
        accommodation = self.get_accommodation()

        user = {'origin': provenance, 'accomodation': accommodation, 'visit_type': visit_type, 'place_pref': place_pref,
                'date': date_of_visit}

        user_data = pd.read_csv('./data/Recommendation data/user_data.csv')
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(user_data[user_data.columns[1:]])

        S = predict_score(kmeans, user_data, user['origin'], user['visit_type'], num_clusters)

        S['place_type'] = S['place_name']
        score_list_S = self.update_score(S, user)
        for index, row in S.iterrows():
            S['score'][index] = score_list_S[index]

        S = S.drop(columns=['place_type'])
        S['score'] = minmax_scale(S['score'])

        df_metrics = pd.read_csv('./data/Forecast Data/dataset_predicted.csv')
        all_metric_score = get_metrics(df_metrics, user)

        rec_dataset = pd.read_csv('./data/Recommendation data/rec_dataset.csv')
        rec_dataset['all_metric_score'] = 0
        rec_dataset['place_score'] = 0
        places_features = extract_places_features(rec_dataset, all_metric_score, user)

        score_list = score_func(user, places_features)
        for index, row in places_features.iterrows():
            places_features['place_score'][index] = (score_list[index] * 10) + (places_features['metric'][index] * 0.00005) + (
                        places_features['all_metric_score'][index] * 0.001)

        dataframe = reshape_df(places_features)

        dataframe1, dataframe2 = merge_dfs(S, dataframe)
        df = pd.concat([dataframe1, dataframe2]).drop_duplicates(keep=False)
        df = df.sort_values(by='place_score', ascending=False)
        df = df.reset_index(drop=True)
        recommendation_results = pd.DataFrame(df.iloc[df.index[0:3]]['place_name'])
        recommendation_results = recommendation_results.set_index('place_name')
        recommendation_results['address'] = ''# recommendation_results.index
        addresses = load_geo_coords()
        for idx in recommendation_results.index:
            recommendation_results['address'][idx] = addresses['address'][idx]
        recommendation_results = recommendation_results.to_dict()
        context['selected_country'] = country
        context['selected_visit'] = visit_type
        context['selected_accommodation'] = accommodation
        context['selected_preference'] = place_pref
        context['date_picker'] = date_of_visit
        context['RecommendationResults'] = recommendation_results['address']

        context['CountriesList'] = CountriesList
        context['GermanCitiesList'] = GermanCitiesList
        context['DistrictList'] = DistrictList
        return context


class ContactView(TemplateView):
    template_name = 'webapp/contact.html'

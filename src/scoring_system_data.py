#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# state : dataframe for munich visitors [rows: months , columns : places]
# listings :  dataframe for listings [rows: months, columns : districts]
# df: tripadvisor dataframe (columns: date, place,rating, #_of_visits, city_district: Maxvorstadt, Hadern...)
# dataframe: (date, place,rating,visit: als paar;mit der Familie ect.. )
# dataset : date, places (metrics= ratings)


outdoors_places = ['Allianz Arena', 'English Garden', 'Eisbach', 'Marienplatz', 'Olympiaturm',
                   'Olympiastadion', 'Olympiapark', 'Tierpark Hellabrunn', 'Viktualienmarkt' ]

indoors_places = ['Alte Pinakothek', 'Asamkirche Munich', 'Bayerisches Nationalmuseum', 'Bayerisches Staatsoper',
                  'BMW Museum', 'Deutsches Museum', 'Kleine Olympiahalle', 'Lenbachhaus', 'Museum Mensch und Natur',
                  'Muenchner Stadtmuseum', 'Muenchner Kammerspiele', 'Munich Residenz', 'Muenchner Philharmoniker',
                  'Museum Brandhorst', 'Nationaltheater', 'Neue Pinakothek', 'Neues Rathaus Munich',
                  'Nymphenburg Palace',
                  'Olympiahalle', 'Olympia-Eissportzentrum',
                  'Prinzregententheater', 'Pinakothek der Moderne',
                  'Schack galerie', 'St-Peter Munich', 'Staatstheater am Gaertnerplatz']
# take the metrics at June 2020 for all places ( from dataset.csv)
all_metric_score = dataset.tail(1)
# df is dataframe taken from data_forecast code
# add a new column with the city_distict_metric
df['city_district_metric'] = df['city_district']
df2 = df
df2['all_metric_score'] = ''
for index, row in df2.iterrows():
    if df2['city_district'][index] != '':
        df2['city_district_metric'][index] = neigh_metric_score.iloc[0][df2['city_district'][index]]
    df2['all_metric_score'][index] = all_metric_score.iloc[0][df2['place'][index]]
# add a new column with the type_door : indoors or outdoors    
df2['type_door'] = df2['place']
for index, row in df2.iterrows():
    for type_door in outdoors_places:
        if df2['place'][index] == type_door:
            df2['type_door'][index] = 'outdoors'
    for type_door in indoors_places:
        if df2['place'][index] == type_door:
            df2['type_door'][index] = 'indoors'
# store the dataset into a csv file for further use
rec_dataset = df2
rec_dataset.to_csv('C:/Users/Oumaima/Documents/AMI/rec_dataset.csv')

# extract the data from the csv file with specific columns
places_features = pd.read_csv('C:/Users/Oumaima/Documents/AMI/rec_dataset.csv',
                              usecols=['date', 'place', 'rating', '#_of_visits', 'city_district',
                                       'city_district_metric', 'all_metric_score', 'type_door'])
places_features = places_features.dropna(subset=['city_district'])

# group by the place to have at the end for each pleace its features : rating , nbr of visits, metrics,type_door
places_features2 = places_features.groupby(by='place').agg(
    {'rating': 'mean', '#_of_visits': 'sum', 'city_district_metric': 'mean', 'all_metric_score': 'mean'})
places_features2.reset_index(level=0, inplace=True)

# The information of city district will get lost with the grouping: so I only repeated what was used for dataset.csv
places_features2['city_district'] = places_features2['place']
city_district = ['Altstadt-Lehel', 'Ludwigsvorstadt-Isarvorstadt', 'Maxvorstadt', 'Schwabing-West'
    , 'Au-Haidhausen', 'Sendling', 'Sendling-Westpark', 'Schwanthalerhöhe'
    , 'Neuhausen-Nymphenburg', 'München-Moosach', 'Milbertshofen-Am Hart', 'Schwabing-Freimann'
    , 'Bogenhausen', 'Berg am Laim', 'Trudering-Riem', 'Ramersdorf-Perlach', 'Obergiesing'
    , 'Untergiesing-Harlaching', 'Thalkirchen-Obersendling-Forstenried-Fürstenried-Solln', 'Hadern'
    , 'Pasing-Obermenzing', 'Aubing-Lochhausen-Langwied', 'Allach-Untermenzing', 'Feldmoching-Hasenbergl'
    , 'Laim']
places = places_features2['place'].unique()
y = {}
geolocator = Nominatim(user_agent='salut', timeout=3)
for place in places:
    try:
        print(place)
        location = geolocator.geocode(place, addressdetails=True, country_codes='de')
        location2 = location.address
        for district in city_district:
            if district in location2:
                y[place] = district
    except:
        y[place] = ''
for index, row in places_features2.iterrows():
    places_features2['city_district'][index] = y[row['place']]

# rearrange it the type door since it will also get lost : any other solution?
places_features2['type_door'] = places_features2['place']
for index, row in places_features2.iterrows():
    for type_door in outdoors_places:
        if places_features2['place'][index] == type_door:
            places_features2['type_door'][index] = 'outdoors'
    for type_door in indoors_places:
        if places_features2['place'][index] == type_door:
            places_features2['type_door'][index] = 'indoors'

# save the places features after the rearrangement in a new dataset
places_features2.to_csv('C:/Users/Oumaima/Documents/AMI/places_features.csv')

# prepare dataframes of the users entries
train_user_entries = [
    {'origin': 'Berlin', 'accomodation': 'Maxvorstadt', 'visit_type': 'alone', 'place_pref': 'indoors'},
    {'origin': 'Prag', 'accomodation': 'Hadern', 'visit_type': 'with_family', 'place_pref': 'outdoors'},
    {'origin': 'Berlin', 'accomodation': 'Hadern', 'visit_type': 'alone', 'place_pref': 'indoors'},
    {'origin': 'Prag', 'accomodation': 'Maxvorstadt', 'visit_type': 'alone', 'place_pref': 'outdoors'}]
test_user_entries = [
    {'origin': 'Bonn', 'accomodation': 'Maxvorstadt', 'visit_type': 'with_family', 'place_pref': 'outdoors'},
    {'origin': 'Prag', 'accomodation': 'Maxvorstadt', 'visit_type': 'alone', 'place_pref': 'outdoors'},
    {'origin': 'Bonn', 'accomodation': 'Hadern', 'visit_type': 'alone', 'place_pref': 'indoors'},
    {'origin': 'Paris', 'accomodation': 'Maxvorstadt', 'visit_type': 'alone', 'place_pref': 'indoors'}]
train_df_entries = pd.DataFrame(train_user_entries)
test_df_entries = pd.DataFrame(test_user_entries)


# define a scoring function basing on the similarities between the places features and user entries
def score_func(user):
    place_score = {}
    for index, row in places_features2.iterrows():
        place_score[index] = 0
        if places_features2['city_district'][index] == user['accomodation']:
            place_score[index] = place_score[index] + 10;
        if places_features2['type_door'][index] == user['place_pref']:
            place_score[index] = place_score[index] + 10;
    return (place_score)


# add another column to pleaces features called place_score: only for demonstration of results
places_features2['place_score'] = places_features2['rating']
for index, row in places_features2.iterrows():
    # the final score = sum of all weighted features + the score of similarities
    places_features2['place_score'][index] = score_func(test_df_entries.iloc[0])[index] * 10 + \
                                             places_features2['rating'][index] + places_features2['#_of_visits'][
                                                 index] * 0.1 + places_features2['all_metric_score'][index] * 50
places_features2.sort_values(by='place_score', ascending=False)

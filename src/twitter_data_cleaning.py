import json
import re

import emot
import geograpy
import nltk
import numpy as np
import pandas as pd
import requests
from flatten_dict import flatten
from geopy.geocoders import Nominatim

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')



def geolocator(coords):
    geolocator = Nominatim(timeout=3)
    location = geolocator.reverse(coords, zoom='10', language='en')
    location = location.address
    return location


def verify_location(places):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if places.country_cities:
        location = ''.join(
            filter(whitelist.__contains__, str(list(places.country_cities.values())[0]))) + ', ' + str(
            list(places.country_cities)[0])
    elif places.cities:
        location = ''.join(
            filter(whitelist.__contains__, str(places.cities)))

    else:
        location = ''.join(
            filter(whitelist.__contains__, str(places.countries)))

    return location


def geo_unify(df):
    n = 0
    res = df.to_records(index=False)
    for row in res:
        if not (np.array(pd.isnull(row['geo\coordinates'])).any()):
            location = geolocator(row['geo\coordinates'])
        elif not (np.array(pd.isnull(row['place\\full_name'])).any()):
            location = row['place\\full_name'] + ', ' + row['place\\country']
        elif not (np.array(pd.isnull(row['user\location'])).any()):
            places = geograpy.get_place_context(text=row['user\location'])
            location = verify_location(places)
        else:
            location = ""
        row['location'] = location
        print(n)
        n = n + 1
        if n == 100:
            break
    df = pd.DataFrame.from_records(res)
    return df


def convert_emotes(df):
    rec = df.to_records(index=False)
    for row in rec:
        for emoticon in emot.EMOTICONS:
            row['full_text'] = re.sub(u'(' + emoticon + ')', "_".join(
                emot.EMOTICONS[emoticon].replace(",", "").split()), row['full_text'])
        for emoj in emot.UNICODE_EMO:
            row['full_text'] = re.sub(u'(' + emoj + ')',
                                      "_".join(emot.UNICODE_EMO[emoj].replace(",", "").replace(":", "").split()),
                                      row['full_text'])
    df = pd.DataFrame.from_records(rec)

    return df


def nltkTokenize(df):
    rec = df.to_records(index=False)
    for row in rec:
        row['full_text'] = nltk.word_tokenize(row['full_text'])
    df = pd.DataFrame.from_records(rec)
    return df


tweets = list()

with open('./data/json/2020-03-20.json', 'r') as fh:
    tweets2 = json.load(fh)
for tweet in tweets2:
    tweet_flat = flatten(tweet, reducer='path')
    tweets.append(tweet_flat)

df = pd.DataFrame.from_records(tweets)
df = df[['id', 'full_text', 'user\location', 'geo\coordinates', 'place\\full_name', 'place\\country']]
df['location'], df['part_of_s'] = ["", ""]

df = geo_unify(df[0:33])
df = df[['id', 'full_text', 'location', 'part_of_s']]
df = convert_emotes(df)
df = nltkTokenize(df)
import json
import re
import json
import requests

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



def remove_tab_newLines_lowercase(df):
    # tweets is a list of tweets
    res = df.to_records(index=False)
    for row in res:
        text = row['full_text']
        text = ' '.join(text.split())
        text = text.lower()
        row['full_text'] = text
    df = pd.DataFrame.from_records(res)
    return df


def replace_slang_words(df):
    error = 'None of the words'
    res = df.to_records(index=False)
    for row in res:
        slangText = row['full_text']
        print(slangText)
        prefixStr = '<div class="translation-text">'
        postfixStr = '</div'
        r = requests.post('https://www.noslang.com/', {'action': 'translate', 'p':
            slangText, 'noswear': 'noswear', 'submit': 'Translate'})
        startIndex = r.text.find(prefixStr) + len(prefixStr)
        endIndex = startIndex + r.text[startIndex:].find(postfixStr)
        if not error in r.text[startIndex:endIndex]:
            row['full_text'] = r.text[startIndex:endIndex]
    df = pd.DataFrame.from_records(res)
    return df


def part_of_speech(df):
    res = df.to_records(index=False)
    for row in res:
        text = row['full_text']
        text = nltk.pos_tag(nltk.word_tokenize(text))
        row['part_of_s'] = text
    df = pd.DataFrame.from_records(res)
    return df

def remove_hashtag(df):
    rec = df.to_records(index=False)
    for row in rec:
        row['full_text'] = row['full_text'].replace('#','')
    df = pd.DataFrame.from_records(rec)
    return df

def remove_stopwords(df):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    rec = df.to_records(index=False)
    for row in rec:
        row['full_text'] = " ".join([w for w in str(row['full_text']).split() if w not in stop_words])
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
df = remove_tab_newLines_lowercase(df)
df = remove_hashtag(df)
df = convert_emotes(df)
df = replace_slang_words(df)
df = remove_stopwords(df)
df = part_of_speech(df)
df = nltkTokenize(df)

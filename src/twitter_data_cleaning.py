import json
import re
import preprocessor as p
import emot
import geograpy
import nltk
import nltk
import numpy as np
import pandas as pd
import requests
from flatten_dict import flatten
from geopy.geocoders import Nominatim
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')


def geolocator(coords):
    """
    Convert the string of latitude and longitude
    coordinates into the corresponding address.

    Uses Nominatim

    :param coords : string
    :return location : string
    """

    geolocator = Nominatim(timeout=3)
    location = geolocator.reverse(coords, zoom='10', language='en')
    location = location.address
    return location


def verify_location(places):
    """
    Extract place information (country and city,
    city only or country only depending on availability)

    :param places : dict
        
    :return location : string
    """

    # White List to filter addresses
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if places.country_cities:
        # Both Country and City information are available
        location = ''.join(
            filter(whitelist.__contains__, str(list(places.country_cities.values())[0]))) + ', ' + str(
            list(places.country_cities)[0])
    elif places.cities:
        # Only City information are available
        location = ''.join(
            filter(whitelist.__contains__, str(places.cities)))
    else:
        # Only Country information are available
        location = ''.join(
            filter(whitelist.__contains__, str(places.countries)))
    return location


def geo_unify(df):
    """
    Check if tweet-related or user-related
    location is available.
    Use the most accurate version (tweet
    location are more accurate than user
    location).

    :param df : DataFrame

    :return df : DataFrame
    """

    # n = 0
    res = df.to_records(index=False)
    for row in res:

        if not (np.array(pd.isnull(row['geo\coordinates'])).any()):
            # If Coordinates are given get the corresponding address
            location = geolocator(row['geo\coordinates'])
        elif not (np.array(pd.isnull(row['place\\full_name'])).any()):
            # Otherwise, get location from place
            location = row['place\\full_name'] + ', ' + row['place\\country']
        elif not (np.array(pd.isnull(row['user\location'])).any()):
            # If no tweet-related location information is available,
            # verify the existence of user defined location and save it
            places = geograpy.get_place_context(text=row['user\location'])
            location = verify_location(places)
        else:
            # If no condition holds give back an empty string
            location = ""
        row['location'] = location
        # print(n)
        # n = n + 1
        # if n == 100:
        #     break
    df = pd.DataFrame.from_records(res)
    return df


def convert_emotes(df):
    """
    Convert emoticons and emojis in tweets
    to their equivalent meaning.

    This was done using the list provided by
    @NeelShah18 and @kakashubham in the emot
    package: https://github.com/NeelShah18/emot
    
    :param df : DataFrame
        
    :return df : Dataframe
    """

    rec = df.to_records(index=False)
    for row in rec:

        for emoticon in emot.EMOTICONS:
            # Replace emoticons found in tweet by their equivalent meaning
            row['full_text'] = re.sub(u'(' + emoticon + ')', "_".join(
                emot.EMOTICONS[emoticon].replace(",", "").split()), row['full_text'])
        for emoj in emot.UNICODE_EMO:
            # Replace emoticons found in tweet by their equivalent meaning
            row['full_text'] = re.sub(u'(' + emoj + ')',
                                      "_".join(emot.UNICODE_EMO[emoj].replace(",", "").replace(":", "").split()),
                                      row['full_text'])
    df = pd.DataFrame.from_records(rec)
    return df


def nltkTokenize(df):
    """
    Tokenize tweets using ntlk's
    word tokenizer
    
    :param df : DataFrame
        
    :return df : ´DataFrame
    """
    rec = df.to_records(index=False)
    for row in rec:
        row['full_text'] = nltk.word_tokenize(row['full_text'])
    df = pd.DataFrame.from_records(rec)
    return df


def remove_tab_newLines_lowercase(df):
    """
    Deletes newlines and tabs and convert uppercase
    letters to lowercase letters.
    Reduce the number of possible characters in the text

    :param df : DataFrame

    :return df : DataFrame
    """
    # tweets is a list of tweets
    res = df.to_records(index=False)
    for row in res:
        text = row['full_text']
        # remove tabs and new lines
        text = ' '.join(text.split())
        # convert to lowercase
        text = text.lower()
        row['full_text'] = text
    df = pd.DataFrame.from_records(res)
    return df


def replace_slang_words(df):
    """
    Replace slang word and spelling mistakes
    with the correct value using the slang
    dictionary website: https://noslang.com

    :param df : DataFrame

    :return df : DataFrame
    """
    error = 'None of the words'
    res = df.to_records(index=False)
    for row in res:
        # use the text slang dictionary "noslang" from noslang.com
        slangText = row['full_text']
        prefixStr = '<div class="translation-text">'
        postfixStr = '</div'
        # make an http post request
        r = requests.post('https://www.noslang.com/', {'action': 'translate', 'p':
            slangText, 'noswear': 'noswear', 'submit': 'Translate'})
        startIndex = r.text.find(prefixStr) + len(prefixStr)
        endIndex = startIndex + r.text[startIndex:].find(postfixStr)
        # check if at least one slang word is unrecognized
        if not error in r.text[startIndex:endIndex]:
            row['full_text'] = r.text[startIndex:endIndex]
    df = pd.DataFrame.from_records(res)
    return df


def part_of_speech(df):
    """
    Part of Speech (PoS) Tagging
    for each word in tweets

    :param df : DataFrame

    :return df : DataFrame
    """

    res = df.to_records(index=False)
    for row in res:
        text = row['full_text']
        text = nltk.pos_tag(text)
        row['part_of_s'] = text
    df = pd.DataFrame.from_records(res)
    return df


def remove_hashtag(df):
    """
    Remove hashtags from tweets

    :param df : DataFrame

    :return df : DataFrame
    """
    rec = df.to_records(index=False)
    for row in rec:
        row['full_text'] = row['full_text'].replace('#', '')
    df = pd.DataFrame.from_records(rec)
    return df


def remove_stopwords(df):
    """
    Remove stop words from tweets

    :param df : DataFrame

    :return df : DataFrame
    """
    stop_words = set(nltk.corpus.stopwords.words('english'))
    rec = df.to_records(index=False)
    for row in rec:
        row['full_text'] = " ".join([w for w in str(row['full_text']).split() if w not in stop_words])
    df = pd.DataFrame.from_records(rec)
    return df


# Function to clean the tweet from URLs, mentions, retweets and unnecessary numbers
def tweet_preprocessing_cleaning(text):
    """
    Clean the tweet from URLs, mentions, retweets and unnecessary numbers
    :param text: string
    :return: string
    """
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.NUMBER, p.OPT.RESERVED)
    return p.clean(text)


def tweet_preprocessing_spelling(text):
    """
    Rectify the spelling of the tweet text

    :param text: string
    :return: string
    """
    spell = SpellChecker()
    tokens = nltk.word_tokenize(text)
    misspelled = spell.unknown(tokens)

    for word in misspelled:
        # Remove adjacent duplicates letters
        xx = remove_duplicates(word)
        # Replace the misspelled word by the most likely one
        text = text.replace(word, spell.correction(xx))
        # Get a list of `likely` options
        # print(spell.candidates(word))

    return text


def tweet_preprocessing_stemming(df):
    """
    Realize the stemming of the words
    :param df: DataFrame
    :return: DataFrame
    """
    stemmer = PorterStemmer()
    res = df.to_records(index=False)

    for row in res:
        words = word_tokenize(row['full_text'])
        for word in words:
            row['full_text'] = row['full_text'].replace(word, stemmer.stem(word))

    df = pd.DataFrame.from_records(res)

    return df


def tweet_preprocessing_lemmatization(df):
    """
    Realize the lemmatization of the words
    :param df: Dataframe
    :return: Dataframe
    """
    lm = WordNetLemmatizer()
    res = df.to_records(index=False)

    for row in res:
        words = word_tokenize(row['full_text'])
        for word in words:
            row['full_text'] = row['full_text'].replace(word, lm.lemmatize(word))

    df = pd.DataFrame.from_records(res)

    return df


# Function to remove adjacent duplicates characters from a string
def remove_duplicates(s):
    chars = list(s)
    prev = None
    k = 0

    for c in s:
        if prev != c:
            chars[k] = c
            prev = c
            k = k + 1

    return ''.join(chars[:k])


def main():
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
    df = nltkTokenize(df)
    df = part_of_speech(df)



if __name__ == "__main__":
    # execute only if run as a script
    main()

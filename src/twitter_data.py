import csv
import json
import os.path
from zipfile import ZipFile

import pandas as pd
import requests as requests
from twarc import Twarc


def get_raw_data_ids(startdate, enddate):
    """
    Download Tweet ID's from the GeoCoV19 Dataset.
    The data is only available from 2020-02-01 to
    2020-05-01.
    For more infomation about the dataset, refer
    to: https://crisisnlp.qcri.org/covid19


    :param startdate: string
    :param enddate: string
    :return:
    """
    dates = pd.date_range(start=startdate, end=enddate)

    for date in dates:
        print('Beginning download for:' + str(date.date()))

        url = "https://crisisnlp.qcri.org/covid_data/ids_files/ids_" + str(date.date()) + ".zip"
        r = requests.get(url)

        # Download zip files with tweet IDs
        with open("./data/tsv_full" + str(date.date()) + ".zip", 'wb') as f:
            f.write(r.content)
        # Extract all the contents of zip file in current directory
        with ZipFile("./data/tsv_full/" + str(date.date()) + ".zip", 'r') as zipObj:
            zipObj.extractall()
        os.remove(zipObj)
        # Rename tsv files according to format: 'yyyy-mm-dd.tsv'
        os.rename('./data/tsv_full/ids_' + str(date.date()) + '.tsv', './data/tsv_full/' + str(date.date()) + '.tsv')


def hydrate_raw_data(t, startdate, enddate, datatype):
    """
    Hydrate raw tweet IDs. The generated tweets are
    filtered according to language and  the existance
    of location information (tweet- and/or user-related
    data) with a limit of 5000 tweet.
    Output is saved in JSON file for tweets and csv files
    for tweet IDS.
    Accepts only tsv and csv files.

    :param t: twarc instance
    :param startdate: string
    :param enddate: string
    :param datatype: string
    :return:
    """

    dates = pd.date_range(start=startdate, end=enddate)

    # For tsv files
    if datatype == 'tsv':
        print('hydrating tweets from ' + str(date.date()) + '...')
        for date in dates:
            numTweets = 0
            ids = []
            ids_filtered = []
            tweets = list()
            with open("./data/tsv_full/" + str(date.date()) + ".tsv") as fd:
                rd = csv.reader(fd, delimiter="\t", quotechar='"', )
                for row in rd:
                    ids.append(row[0])
            # Hydrate tweet IDs
            tweet_hy = t.hydrate(ids)
            with open('./data/json/' + str(date.date()) + ".json", 'w') as outfile:
                for tweet in tweet_hy:
                    newRow = {}
                    # Keep only tweets written in english and with some location information
                    if ((tweet.get('lang') == 'en') and (tweet.get('geo') or tweet.get('coordinates')
                                                         or tweet.get('place') or tweet.get('user').get(
                                'location'))):
                        for key, value in tweet.items():
                            newRow[key] = value
                        tweets.append(newRow)
                        if numTweets % 100 == 0:
                            print('Tweets collected so far: {:d}'.format(numTweets))
                        numTweets = numTweets + 1
                        ids_filtered.append(tweet.get('id'))
                    # Stop collecting tweets after 5000 Tweet
                    if numTweets == 5000:
                        break
                ids_filtered_df = pd.DataFrame(ids_filtered)
                print('generating json file in: ./data/json/' + str(date.date()) + ".json")
                # Save IDs of collected tweets in csv files
                ids_filtered_df.to_csv(r'./data/csv/' + str(date.date()) + ".csv", index=False, header=True)
                # Save tweets in JSON files
                outfile.write(json.dumps(tweets))

    # For csv files
    elif datatype == 'csv':
        for date in dates:
            print('hydrating tweets from ' + str(date.date()) + '...')
            numTweets = 0
            tweets = list()
            ids = []
            ids_geo = []
            ids_filtered = []
            # Search for Files with geo coordinates
            if os.path.exists(r"./data/csv_geo/" + str(date.date()) + ".csv"):
                with open(r"./data/csv_geo/" + str(date.date()) + ".csv") as fd:
                    rd = csv.reader(fd, quotechar='"', )
                    for row in rd:
                        ids.append(row[0])
                        ids_geo.append(row[0])
            with open(r"./data/csv_full/" + str(date.date()) + ".csv") as fd:
                rd = csv.reader(fd, quotechar='"', )
                for row in rd:
                    if not (row[0] in ids_geo):
                        ids.append(row[0])
            # Hydrate tweet IDs
            tweet_hy = t.hydrate(ids)

            with open(r'./data/json/' + str(date.date()) + ".json", 'w') as outfile:
                for tweet in tweet_hy:
                    newRow = {}
                    # Keep only tweets written in english and with some location information
                    if (tweet.get('lang') == 'en' and (tweet.get('geo') or tweet.get('coordinates')
                                                       or tweet.get('place') or tweet.get('user').get('location'))):
                        for key, value in tweet.items():
                            newRow[key] = value
                        tweets.append(newRow)
                        if numTweets % 100 == 0:
                            print('Tweets collected so far: {:d}'.format(numTweets))
                        numTweets = numTweets + 1
                        ids_filtered.append(tweet.get('id'))
                    # Stop collecting tweets after 5000 Tweet
                    if numTweets == 5000:
                        break
                ids_filtered_df = pd.DataFrame(ids_filtered)
                print('generating json file in: ./data/json/' + str(date.date()) + ".json")
                # Save IDs of collected tweets in csv files
                ids_filtered_df.to_csv(r'./data/csv/' + str(date.date()) + ".csv", index=False, header=True)
                # Save tweets in JSON files
                outfile.write(json.dumps(tweets))
    # For other data types
    else:
        print('Data Type is not valid')


def get_json_data(date):
    """
    Read JSON file for the given date
    and save content to dict.


    :param date: string
    :return: tweets_dict: dict
    """
    with open('./data/json/' + date + ".json", 'r') as fh:
        tweets_dict = json.load(fh)
        return tweets_dict


def hydrate_data(t, startdate, enddate):
    """
    Hydrate tweet IDs and collect tweet
    information.
    Output is a JSON file with tweets data
    Accepts only tsv and csv files.

    :param t: twarc instance
    :param startdate: string
    :param enddate: string
    :return:
    """
    dates = pd.date_range(start=startdate, end=enddate)

    for date in dates:
        print('hydrating tweets from ' + str(date.date()) + '...')
        numTweets = 0
        tweets = list()
        ids = []
        with open(r"../data/emotion_detection_data/csv/" + str(date.date()) + ".csv") as fd:
            rd = csv.reader(fd, quotechar='"', )
            header = next(rd)
            for row in rd:
                ids.append(row[0])
        # Hydrate tweet IDs
        tweet_hy = t.hydrate(ids)
        # Start collecting tweets
        with open(r'../data/emotion_detection_data/json/' + str(date.date()) + ".json", 'w') as outfile:
            for tweet in tweet_hy:
                newRow = {}
                for key, value in tweet.items():
                    newRow[key] = value
                tweets.append(newRow)
                if numTweets % 100 == 0:
                    print('Tweets collected so far: {:d}'.format(numTweets))
                numTweets = numTweets + 1
            print('generating json file in: ../data/emotion_detection_data/json/' + str(date.date()) + ".json")
            # Save tweets in JSON files
            outfile.write(json.dumps(tweets))


def main():
    # Twitter Api Keys, if access is required
    # please contact: salem.sfaxi@tum.de for
    # credentials
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''
    t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)

    hydrate_data(t, startdate='', enddate='')
    tweets_dict = get_json_data('')


if __name__ == "__main__":
    # execute only if run as a script
    main()

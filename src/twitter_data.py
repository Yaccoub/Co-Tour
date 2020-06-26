from zipfile import ZipFile
import os.path

import requests as requests
import csv
import json

import pandas as pd
from twarc import Twarc


def get_raw_data_ids(startdate, enddate):
    dates = pd.date_range(start=startdate, end=enddate)

    for date in dates:
        print('Beginning download for:' + str(date.date()))

        url = "https://crisisnlp.qcri.org/covid_data/ids_files/ids_" + str(date.date()) + ".zip"
        r = requests.get(url)

        with open("./data/tsv_full" + str(date.date()) + ".zip", 'wb') as f:
            f.write(r.content)
        with ZipFile("./data/tsv_full/" + str(date.date()) + ".zip", 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall()
        os.remove(zipObj)
        os.rename('./data/tsv_full/ids_' + str(date.date()) + '.tsv', './data/tsv_full/' + str(date.date())+'.tsv')

def hydrate_raw_data(t, startdate, enddate, datatype):

    dates = pd.date_range(start=startdate, end=enddate)

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

            tweet_hy = t.hydrate(ids)
            with open('./data/json/' + str(date.date()) + ".json", 'w') as outfile:
                for tweet in tweet_hy:
                    newRow = {}
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
                    if numTweets == 5000:
                        break
                ids_filtered_df = pd.DataFrame(ids_filtered)
                print('generating json file in: ./data/json/' + str(date.date()) + ".json")
                ids_filtered_df.to_csv(r'./data/csv/' + str(date.date()) + ".csv", index=False, header=True)
                outfile.write(json.dumps(tweets))

    elif datatype == 'csv':
        for date in dates:
            print('hydrating tweets from ' + str(date.date()) + '...')
            numTweets = 0
            tweets = list()
            ids = []
            ids_geo = []
            ids_filtered = []
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
            tweet_hy = t.hydrate(ids)

            with open(r'./data/json/' + str(date.date()) + ".json", 'w') as outfile:
                for tweet in tweet_hy:

                    newRow = {}
                    if (tweet.get('lang') == 'en' and (tweet.get('geo') or tweet.get('coordinates')
                                                       or tweet.get('place') or tweet.get('user').get('location'))):
                        for key, value in tweet.items():
                            newRow[key] = value
                        tweets.append(newRow)
                        if numTweets % 100 == 0:
                            print('Tweets collected so far: {:d}'.format(numTweets))
                        numTweets = numTweets + 1
                        ids_filtered.append(tweet.get('id'))

                    if numTweets == 5000:
                        break
                ids_filtered_df = pd.DataFrame(ids_filtered)
                print('generating json file in: ./data/json/' + str(date.date()) + ".json")
                ids_filtered_df.to_csv(r'./data/csv/' + str(date.date()) + ".csv", index=False, header=True)
                outfile.write(json.dumps(tweets))
    else:
        print('Data Type is not valid')

def get_json_data(date):

    with open('./data/json/' + date + ".json", 'r') as fh:
        tweets_dict = json.load(fh)
        return tweets_dict

def hydrate_data(t, startdate, enddate):

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
        tweet_hy = t.hydrate(ids)

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
            outfile.write(json.dumps(tweets))

def main():

    consumer_key = 'FdsFxDo1Bz2qr9bjcbqut1Vom'
    consumer_secret = 'uSpOf5iVrm64lxprVv2WYCD6lWCvVZGGVSdHK4CaO7ZIflFcwq'
    access_token = '1640697636-y90tCiaX5ZZaJEtXlXWndkBiC6vn1V1M15q7uDI'
    access_token_secret = 'SHZ8r8NcEpNQ85ArCDCjIIOP8Z2n1PkWDJQENqH1lM1Xq'
    t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)

    # hydrate_data(t, startdate = '2020-06-20', enddate = '2020-06-20')
    tweetsdict = get_json_data('2020-06-20')


if __name__ == "__main__":
    # execute only if run as a script
    main()
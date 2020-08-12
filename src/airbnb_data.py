#!/usr/bin/env python
# coding: utf-8

# In[131]:

# Import the standard libraries
import pandas as pd
import numpy as np
import requests
import datetime

def download_data(month_listings):
    if month_listings == 'Mar20_listings':
        Airbnb_Mar_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-03-19/visualisations/listings.csv'
        Airbnb_Mar_listings_req = requests.get(Airbnb_Mar_listings_url)
        Airbnb_Mar_listings_url_content = Airbnb_Mar_listings_req.content
        # save the data content in a csv file
        Airbnb_Mar_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichMar20.csv', 'wb')
        Airbnb_Mar_listings_csv_file.write(Airbnb_Mar_listings_url_content)
        Airbnb_Mar_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichMar20.csv')
    
    elif month_listings == 'Mar19_listings':
        Airbnb_Mar_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-03-15/visualisations/listings.csv'
        Airbnb_Mar_listings_req = requests.get(Airbnb_Mar_listings_url)
        Airbnb_Mar_listings_url_content = Airbnb_Mar_listings_req.content
        # save the data content in a csv file
        Airbnb_Mar_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichMar19.csv', 'wb')
        Airbnb_Mar_listings_csv_file.write(Airbnb_Mar_listings_url_content)
        Airbnb_Mar_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichMar19.csv')
        
    elif month_listings == 'Apr20_listings':
        Airbnb_Apr_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-04-25/visualisations/listings.csv'
        Airbnb_Apr_listings_req = requests.get(Airbnb_Apr_listings_url)
        Airbnb_Apr_listings_url_content = Airbnb_Apr_listings_req.content
        # save the data content in a csv file
        Airbnb_Apr_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichApr20.csv', 'wb')
        Airbnb_Apr_listings_csv_file.write(Airbnb_Apr_listings_url_content)
        Airbnb_Apr_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichApr20.csv')

    elif month_listings == 'Mai20_listings':
        Airbnb_Mai_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-05-24/visualisations/listings.csv'
        Airbnb_Mai_listings_req = requests.get(Airbnb_Mai_listings_url)
        Airbnb_Mai_listings_url_content = Airbnb_Mai_listings_req.content
        # save the data content in a csv file
        Airbnb_Mai_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichMai20.csv', 'wb')
        Airbnb_Mai_listings_csv_file.write(Airbnb_Mai_listings_url_content)
        Airbnb_Mai_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichMai20.csv')
    
    elif month_listings == 'Mai19_listings':
        Airbnb_Mai_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-05-22/visualisations/listings.csv'
        Airbnb_Mai_listings_req = requests.get(Airbnb_Mai_listings_url)
        Airbnb_Mai_listings_url_content = Airbnb_Mai_listings_req.content
        # save the data content in a csv file
        Airbnb_Mai_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichMai19.csv', 'wb')
        Airbnb_Mai_listings_csv_file.write(Airbnb_Mai_listings_url_content)
        Airbnb_Mai_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichMai19.csv')

    elif month_listings == 'June20_listings':
        Airbnb_Jun_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-06-20/visualisations/listings.csv'
        Airbnb_Jun_listings_req = requests.get(Airbnb_Jun_listings_url)
        Airbnb_Jun_listings_url_content = Airbnb_Jun_listings_req.content
        # save the data content in a csv file
        Airbnb_Jun_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichJun20.csv', 'wb')
        Airbnb_Jun_listings_csv_file.write(Airbnb_Jun_listings_url_content)
        Airbnb_Jun_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichJun20.csv')
    
    elif month_listings == 'June19_listings':
        Airbnb_Jun_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-06-24/visualisations/listings.csv'
        Airbnb_Jun_listings_req = requests.get(Airbnb_Jun_listings_url)
        Airbnb_Jun_listings_url_content = Airbnb_Jun_listings_req.content
        # save the data content in a csv file
        Airbnb_Jun_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichJun19.csv', 'wb')
        Airbnb_Jun_listings_csv_file.write(Airbnb_Jun_listings_url_content)
        Airbnb_Jun_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichJun19.csv')
    
    elif month_listings == 'July19_listings':
        Airbnb_July_listings_url = ' http://data.insideairbnb.com/germany/bv/munich/2019-07-16/visualisations/listings.csv'
        Airbnb_July_listings_req = requests.get(Airbnb_July_listings_url)
        Airbnb_July_listings_url_content = Airbnb_July_listings_req.content
        # save the data content in a csv file
        Airbnb_July_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichJuly19.csv', 'wb')
        Airbnb_July_listings_csv_file.write(Airbnb_July_listings_url_content)
        Airbnb_July_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichJuly19.csv')
    
    elif month_listings == 'Aug19_listings':
        Airbnb_Aug_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-08-24/visualisations/listings.csv'
        Airbnb_Aug_listings_req = requests.get(Airbnb_Aug_listings_url)
        Airbnb_Aug_listings_url_content = Airbnb_Aug_listings_req.content
        # save the data content in a csv file
        Airbnb_Aug_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichAug19.csv', 'wb')
        Airbnb_Aug_listings_csv_file.write(Airbnb_Aug_listings_url_content)
        Airbnb_Aug_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichAug19.csv')
    
    elif month_listings == 'Sep19_listings':
        Airbnb_Sep_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-09-24/visualisations/listings.csv'
        Airbnb_Sep_listings_req = requests.get(Airbnb_Sep_listings_url)
        Airbnb_Sep_listings_url_content = Airbnb_Sep_listings_req.content
        # save the data content in a csv file
        Airbnb_Sep_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichSep19.csv', 'wb')
        Airbnb_Sep_listings_csv_file.write(Airbnb_Sep_listings_url_content)
        Airbnb_Sep_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichSep19.csv')
    
    elif month_listings == 'Oct19_listings':
        Airbnb_Oct_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-10-20/visualisations/listings.csv'
        Airbnb_Oct_listings_req = requests.get(Airbnb_Oct_listings_url)
        Airbnb_Oct_listings_url_content = Airbnb_Oct_listings_req.content
        # save the data content in a csv file
        Airbnb_Oct_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichOct19.csv', 'wb')
        Airbnb_Oct_listings_csv_file.write(Airbnb_Oct_listings_url_content)
        Airbnb_Oct_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichOct19.csv')

    elif month_listings == 'Nov19_listings':
        Airbnb_Nov_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-11-25/visualisations/listings.csv'
        Airbnb_Nov_listings_req = requests.get(Airbnb_Nov_listings_url)
        Airbnb_Nov_listings_url_content = Airbnb_Nov_listings_req.content
        # save the data content in a csv file
        Airbnb_Nov_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichNov19.csv', 'wb')
        Airbnb_Nov_listings_csv_file.write(Airbnb_Nov_listings_url_content)
        Airbnb_Nov_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichNov19.csv')
    
    elif month_listings == 'Dec19_listings':
        Airbnb_Dec_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2019-12-26/visualisations/listings.csv'
        Airbnb_Dec_listings_req = requests.get(Airbnb_Dec_listings_url)
        Airbnb_Dec_listings_url_content = Airbnb_Dec_listings_req.content
        # save the data content in a csv file
        Airbnb_Dec_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichDec19.csv', 'wb')
        Airbnb_Dec_listings_csv_file.write(Airbnb_Dec_listings_url_content)
        Airbnb_Dec_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichDec19.csv')
    
    elif month_listings == 'Jan20_listings':
        Airbnb_Jan_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-01-22/visualisations/listings.csv'
        Airbnb_Jan_listings_req = requests.get(Airbnb_Jan_listings_url)
        Airbnb_Jan_listings_url_content = Airbnb_Jan_listings_req.content
        # save the data content in a csv file
        Airbnb_Jan_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichJan20.csv', 'wb')
        Airbnb_Jan_listings_csv_file.write(Airbnb_Jan_listings_url_content)
        Airbnb_Jan_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichJan20.csv')
    
    elif month_listings == 'Feb20_listings':
        Airbnb_Feb_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-02-27/visualisations/listings.csv'
        Airbnb_Feb_listings_req = requests.get(Airbnb_Feb_listings_url)
        Airbnb_Feb_listings_url_content = Airbnb_Feb_listings_req.content
        # save the data content in a csv file
        Airbnb_Feb_listings_csv_file = open('../data/Airbnb_data/' + 'listingsMunichFeb20.csv', 'wb')
        Airbnb_Feb_listings_csv_file.write(Airbnb_Feb_listings_url_content)
        Airbnb_Feb_listings_csv_file.close()
        # read from the dowloaded csv file
        return pd.read_csv('../data/Airbnb_data/' + 'listingsMunichFeb20.csv')
    else :
        return 'invalid month'

def clean_data(data):
    rec_data = data.loc[:,[ 'neighbourhood','id','number_of_reviews','room_type','minimum_nights','price','last_review']]
    # rename the number of reviews column for an intuitive use
    rec_data.rename(columns={'number_of_reviews':'reviews_count'}, inplace=True)
    # get rid of the missing values 
    rec_data = rec_data[rec_data['reviews_count'] != 0] 

    # These columns are useful for the prediction system
    # list the data according to the neighbourhoods ; use only columns of : id / reviews_count/ price 
    pred_data = rec_data.groupby(by = 'neighbourhood').agg({'id':'count', 'reviews_count':'sum','price':'mean'}).sort_values(by = 'price', ascending = True)
    # rename the columns
    pred_data.rename(columns={'id':'accomodations_count','price':'avg_price'}, inplace=True)
    return pred_data

def convert_time(data):
    # Change format of date to datetime
    for index in data.index:
        item = str(data.Datum.loc[index])
        item = item[:-9]
        # Rename some month
        if item[:-2] == 'Mai':
            item = list(item)
            item[:-2] = list('May')
            item = "".join(item)
        if item[:-2] == 'June':
            item = list(item)
            item[:-2] = list('Jun')
            item = "".join(item)
        if item[:-2] == 'July':
            item = list(item)
            item[:-2] = list('Jul')
            item = "".join(item)
        data.Datum.loc[index] = '01' + '/' + item[:-2] + '/' + '20' + item[-2:]
        data.Datum.loc[index] = datetime.datetime.strptime(data.Datum.loc[index],'%d/%b/%Y').strftime('%d/%b/%Y')
    return data

def main():
    months_listings = ['Mar19_listings','Mai19_listings','June19_listings','July19_listings','Aug19_listings','Sep19_listings','Oct19_listings','Nov19_listings','Dec19_listings','Jan20_listings','Feb20_listings','Mar20_listings','Apr20_listings','Mai20_listings','June20_listings']
    ret = []
    concat_dfs = []
    i=1
    for month_sheet in months_listings:
        print('Downloading data for: ' + month_sheet )
        # Data download
        df = download_data(month_sheet)
        # the data of the current month listing
        df = clean_data(df)
        # how to concatenate it with the other months listings in a table 
        df[month_sheet] = df.values.tolist()
        df = pd.DataFrame(df[month_sheet])
        ret.append(df)

    # Concat the list to one table
    df = pd.concat(ret, axis=1, sort=True).T.reset_index().rename(columns={"index": "Datum"})
    df = convert_time(df)
    df.to_csv('../data/Airbnb_data/listings.csv',index=False)

if __name__ == '__main__':
    main()

# In[ ]:

#!/usr/bin/env python
# coding: utf-8

# In[62]:


import requests
import pandas as pd
import numpy as np
from datetime import datetime


# In[67]:


# the last complied data was at 20/06 : this should be taken in consideration while using the dataset url
# read the data using requests module
Airbnb_listings_url = 'http://data.insideairbnb.com/germany/bv/munich/2020-06-20/visualisations/listings.csv'
Airbnb_listings_req = requests.get(Airbnb_listings_url)
Airbnb_listings_url_content = Airbnb_listings_req.content
# save the data content in a csv file
Airbnb_listings_csv_file = open('listingsMunich.csv', 'wb')
Airbnb_listings_csv_file.write(Airbnb_listings_url_content)
Airbnb_listings_csv_file.close()

# read from the dowloaded csv file
Airbnb_listings = pd.read_csv('listingsMunich.csv')

# these columns are useful for the recommendation system
Recommendation_Listings = Airbnb_listings.loc[:,[ 'neighbourhood','id','number_of_reviews','room_type','minimum_nights','price','last_review']]
# rename the number of reviews column for an intuitive use
Recommendation_Listings.rename(columns={'number_of_reviews':'reviews_count'}, inplace=True)
# get rid of the missing values 
Recommendation_Listings = Recommendation_Listings[Recommendation_Listings['reviews_count'] != 0] 
# sort the rows according to the timeline
Recommendation_Listings['last_review'] = pd.to_datetime(Recommendation_Listings['last_review'])
Recommendation_Listings = Recommendation_Listings.sort_values(by = 'last_review', ascending = True)

# These columns are useful for the prediction system
# list the data according to the neighbourhoods ; use only columns of : id / reviews_count/ price 
Prediction_Listings = Recommendation_Listings.groupby(by = 'neighbourhood').agg({'id':'count', 'reviews_count':'sum','price':'mean'}).sort_values(by = 'price', ascending = True)
# rename the columns
Prediction_Listings.rename(columns={'id':'accomodations_count','price':'avg_price'}, inplace=True)


# In[ ]:





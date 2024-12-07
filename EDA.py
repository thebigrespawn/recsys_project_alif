#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:43:01 2024

@author: khondamiranvarov
"""

import pandas as pd
from collections import Counter

# Specify the path to your business.json file
file_path_business = './data/yelp_dataset/yelp_academic_dataset_business.json'

business_df = pd.read_json(file_path_business, lines=True)[pd.read_json(file_path_business, lines=True)['categories'].str.contains("Restaurants", na=False)]
business_ids = business_df['business_id'].unique()
#%%
file_path_review = './data/yelp_dataset/yelp_academic_dataset_review.json'

review_df = pd.read_json(file_path_review, lines=True)[pd.read_json(file_path_review, lines=True)['business_id'].isin(business_ids)]



#%%
import pandas as pd

# Specify the path to your business.json file
file_path_business = './data/yelp_dataset/yelp_academic_dataset_business.json'

# Read the business data once
business_df = pd.read_json(file_path_business, lines=True)

# Filter businesses that are restaurants
business_df = business_df[business_df['categories'].str.contains("Restaurants", na=False)]

# Get the set of business_ids of restaurants for faster lookup
business_ids_set = set(business_df['business_id'])

# Specify the path to your review.json file
file_path_review = './data/yelp_dataset/yelp_academic_dataset_review.json'

# Initialize an empty list to store filtered reviews
filtered_reviews = []

# Define the chunk size based on your system's memory capacity
chunk_size = 100000  # Adjust this number as needed

# Read the review data in chunks
for chunk in pd.read_json(file_path_review, lines=True, chunksize=chunk_size):
    # Filter reviews where business_id is in the set of restaurant business_ids
    filtered_chunk = chunk[chunk['business_id'].isin(business_ids_set)]
    filtered_reviews.append(filtered_chunk)

# Concatenate all filtered chunks into a single DataFrame
review_df = pd.concat(filtered_reviews, ignore_index=True)

#%%
print(review_df.head())


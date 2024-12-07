#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:45:26 2024

@author: khondamiranvarov
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from geopy.point import Point
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import folium

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

# Combine categories and attributes into a single text field
business_df['combined_features'] = business_df['categories'].fillna('') + ' ' + business_df['attributes'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Text preprocessing
business_df['combined_features'] = business_df['combined_features'].str.lower()
business_df['combined_features'] = business_df['combined_features'].str.replace(r'[^\w\s]', ' ')  # Remove punctuation





#%%
z = business_df['combined_features']



#%%
import re
import ast
import json
import pandas as pd

def try_parse_dict(value):
    # If value is already a dict, just return it
    if isinstance(value, dict):
        return value

    # If value is None or empty, return empty dict
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == '':
        return {}

    # Convert to string
    s = str(value).strip()

    # Replace booleans/none to Python equivalents for literal_eval
    s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnone\b", "None", s, flags=re.IGNORECASE)

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return parsed
        else:
            return {}
    except:
        # If literal_eval fails, try JSON approach
        json_str = s.replace("'", '"')  
        try:
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, dict):
                return parsed_json
            else:
                return {}
        except:
            return {}

def flatten_nested_dict(d):
    flattened = {}
    for key, value in d.items():
        if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
            nested = try_parse_dict(value)
            for nk, nv in nested.items():
                flattened[f"{key}_{nk}"] = nv
        else:
            flattened[key] = value
    return flattened

def clean_attributes(attributes):
    cleaned = {}
    for k, v in attributes.items():
        # Skip values that are None, False, or their string equivalents
        if v is None:
            continue
        if v is False:
            continue
        if isinstance(v, str) and v.strip().lower() in ['false', 'none', '']:
            continue
        # Otherwise, keep it
        cleaned[k] = v
    return cleaned

def parse_and_flatten_attributes(attr_value):
    parsed = try_parse_dict(attr_value)
    flattened = flatten_nested_dict(parsed)
    cleaned = clean_attributes(flattened)
    return cleaned

def combine_features(row):
    categories_str = row['categories'].lower() if pd.notna(row['categories']) else ''
    attributes_str = ' '.join([f"{k} {str(v).lower()}" for k, v in row['flattened_attributes'].items()])
    combined = f"{categories_str} {attributes_str}".strip()
    return combined

# Example usage (replace with your actual DataFrame)
# business_df = pd.read_json('yelp_academic_dataset_business.json', lines=True)
# Ensure business_df has 'categories' and 'attributes' columns

business_df['flattened_attributes'] = business_df['attributes'].apply(parse_and_flatten_attributes)
business_df['combined_features'] = business_df.apply(combine_features, axis=1)

# Remove punctuation and ensure lowercase
business_df['combined_features'] = business_df['combined_features'].str.replace(r'[^\w\s]', ' ', regex=True)
business_df['combined_features'] = business_df['combined_features'].str.lower()

print(business_df[['categories', 'flattened_attributes', 'combined_features']].head(10))

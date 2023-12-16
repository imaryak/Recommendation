#import libraries 
import pandas as pd
import numpy as np
import math
from numpy.linalg import norm

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from pickle import dump, load

pd.set_option('display.max_columns', None)

np.random_state = 42


# load in df with the cluster labels
sd_pp = pd.read_csv('data/sd_pp', index_col = 0)



# load in df with the listing urls
sd_listings_url = pd.read_csv('data/sd_modeling_with_urls', index_col= 0)
sd_listings_url = sd_listings_url[['listing_url']]
sd_listings_url = sd_listings_url.reset_index(drop = 'index')

# save as csv
path = "data/"

sd_listings_url.to_csv(path + 'url_listings')

# selects a random listing from sd_clustered
random_listing = sd_pp.sample(n = 1)

#random_listing

# convert single listing to an array
random_listing_array = random_listing.values

# convert all listings to an array
sd_pp_array = sd_pp.values

# define two lists or arrays to compare
A = np.squeeze(np.asarray(sd_pp_array))
B = np.squeeze(np.asarray(random_listing_array))
print("A:\n", A)
print("B:\n", B)
 
# compute cosine similarity
cosine = np.dot(A,B)/(norm(A, axis=1)*norm(B))
print("Cosine Similarity:\n", cosine)



# load in df with the cluster labels
sd_clustered = pd.read_csv('data/sd_clustered', index_col = 0)
sd_clustered.reset_index(drop = 'index')

# merge on index
sd_clustered = sd_clustered.join(sd_listings_url)

# add similarity 
rec = sd_clustered.copy()
rec['similarity'] = pd.DataFrame(cosine).values

# reorder column names
rec = rec[['id','listing_url', 'similarity', 'cluster_label', 'latitude', 'longitude',
       'neighbourhood_cleansed', 'zipcode', 'property_type', 'room_type',
       'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
       'nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee',
       'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
       'host_response_time', 'host_response_rate', 'host_is_superhost',
       'host_total_listings_count', 'host_has_profile_pic',
       'host_identity_verified', 'number_of_reviews', 'number_of_stays',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'requires_license', 'instant_bookable',
       'is_business_travel_ready', 'cancellation_policy',
       'require_guest_profile_picture', 'require_guest_phone_verification']]

#print(rec)

# sort by highest similarity
rec.sort_values(by = ['similarity'], ascending = False).head(6)


# selects a random listing from sd_clustered
random_listing = sd_pp.sample(n = 1)
#random_listing

def get_recommendations(df, listing):
    """
    Takes in preprocessed dataframe and selected listing as inputs and gives top 5 (including listing)
    recommendations based on cosine similarity. 
    """
    # reset the index
    df = df.reset_index(drop = 'index')
    
    # convert single listing to an array
    listing_array = listing.values

    # convert all listings to an array
    df_array = df.values
    
    # get arrays into a single dimension
    A = np.squeeze(np.asarray(df_array))
    B = np.squeeze(np.asarray(listing_array))
    
    # compute cosine similarity 
    cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
    # add similarity into recommendations df and reset the index
    rec = sd_clustered.copy().reset_index(drop = 'index')
    rec['similarity'] = pd.DataFrame(cosine).values
    
    # reorder column names
    rec = rec[['id','listing_url', 'similarity', 'cluster_label', 'latitude', 'longitude',
       'neighbourhood_cleansed', 'zipcode', 'property_type', 'room_type',
       'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
       'nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee',
       'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
       'host_response_time', 'host_response_rate', 'host_is_superhost',
       'host_total_listings_count', 'host_has_profile_pic',
       'host_identity_verified', 'number_of_reviews', 'number_of_stays',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'requires_license', 'instant_bookable',
       'is_business_travel_ready', 'cancellation_policy',
       'require_guest_profile_picture', 'require_guest_phone_verification']]

    # x_df=rec[['listing_url','similarity']].sort_values(by = ['similarity'], ascending = False).head(1)


    # # Assuming x_df is a DataFrame with 'listing_url' column
    # selected_listing_url = x_df['listing_url'].values[0]

    # # Use st.components.v1.iframe directly with the extracted URL
    # st.components.v1.iframe(selected_listing_url, width=800, height=600, scrolling=True)



    # st.components.v1.iframe("https://www.airbnb.com/rooms/34364019", width=800, height=600, scrolling=True)

    
    # sort by top 5 descending
    return rec.sort_values(by = ['similarity'], ascending = False).head(6)



# load in df with the cluster labels
sd_trans = pd.read_csv('data/sd_trans', index_col = 0)
#sd_trans.head()

#sd_trans.columns

features_to_keep = ['neighbourhood_cleansed', 'property_type', 'room_type',
                   'accommodates', 'bathrooms', 'beds', 'nightly_price', 'review_scores_rating']

sd_simplified = sd_trans[features_to_keep]
#sd_simplified.head()

# save as csv
path = "notebooks/"

sd_simplified.to_csv(path + 'sd_simplified')

# define nominal and ordinal features in the categorical columns
nom_cols = sd_simplified.select_dtypes(['object']).columns
#print(nom_cols)

ordinal_cols = sd_simplified.select_dtypes(['category']).columns
#print(ordinal_cols)

# define numeric transformation pipeline that scales the numbers
numeric_pipeline = Pipeline([('numnorm', StandardScaler())]) 


# define a nominal transformation pipeline that OHE the cats
nominal_pipeline = Pipeline([('onehotenc', OneHotEncoder(categories= "auto", 
                                                         sparse = False, 
                                                         handle_unknown = 'ignore'))]) 
# construct column transformer for the selected columns with pipelines
ct = ColumnTransformer(transformers = [("nominalpipe", nominal_pipeline, ['neighbourhood_cleansed', 
                                                                          'property_type', 'room_type']),
                                       ("numericpipe", numeric_pipeline, sd_simplified.select_dtypes(['int', 'float']).columns)])

# save the NEW column transformer
dump(ct, open('notebooks/simple_column_transformer.pkl', 'wb'))

sd_simplified_pp = pd.DataFrame(ct.fit_transform(sd_simplified))

#print(sd_simplified_pp.head())

import pandas as pd

# List of columns you want in the input DataFrame
columns_for_input = ['neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'beds', 'nightly_price', 'review_scores_rating']

# Create an empty DataFrame with the specified columns
input_df = pd.DataFrame(columns=columns_for_input)

# Display the DataFrame
input_df.head()

# create an empty df with news column for input of values for recommendations
input_df = pd.DataFrame(columns = sd_simplified.columns)
import streamlit as st


# if st.session_state.selected_option == 0:
    


def get_simplified_recommendations(df, listing):
    """
    Takes in preprocessed dataframe and preprocessed custom listing as inputs and gives top 5
    recommendations based on cosine similarity. 
    """
    # reset the index
    df = df.reset_index(drop = 'index')
    
    # preprocess the listing input
    listing_pp = pd.DataFrame(ct.transform(listing))
    
    # convert single listing to an array
    listing_array = listing_pp.values
    # convert all listings to an array
    df_array = df.values
    
    # get arrays into a single dimension
    A = np.squeeze(np.asarray(df_array))
    B = np.squeeze(np.asarray(listing_array))
    
    # compute cosine similarity 
    cosine = np.dot(A,B)/(norm(A, axis = 1)*norm(B))
    
    # add similarity into recommendations df and reset the index
    rec = sd_simplified.copy().reset_index(drop = 'index')
    rec['similarity'] = pd.DataFrame(cosine).values
    
    # add in listings_urls
    # merge on index
    rec = rec.join(sd_listings_url)
    
    # reorder column names
    rec = rec[['listing_url', 'similarity', 'neighbourhood_cleansed', 'property_type', 
               'room_type', 'accommodates', 'bathrooms', 'beds', 'nightly_price', 'review_scores_rating']]
    
    # sort by top 5 descending
    return rec.sort_values(by = ['similarity'], ascending = False).head(5)


# get top 5 most similar listings to user input

import streamlit as st
selected_option = st.selectbox("Select an Action", ["Get random recommendation", "Insert data and recommend"])

# Check the selected option and execute the corresponding function
if selected_option == "Get random recommendation":
    
    st.write((get_recommendations(sd_pp, random_listing)))

elif selected_option == "Insert data and recommend":
    
    neighborhood_input = st.text_input("Enter Neighborhood", "La Jolla")
    property_input = st.text_input("Enter Property Type", "House")
    room_input = st.text_input("Enter Room Type", "Entire home/apt")
    accommodates = st.number_input("Enter Number of Accommodates", 1, 10, 3, 1)
    bathrooms = st.number_input("Enter Number of Bathrooms", 0.5, 10.0, 2.0, 0.5)
    beds = st.number_input("Enter Number of Beds", 1, 10, 2, 1)
    price = st.number_input("Enter Price per Night", 1.0, 1000.0, 250.0, 1.0)
    rating = st.number_input("Enter Rating", 0.0, 100.0, 90.0, 1.0)




# Assign input values to input_df columns
    input_df = pd.DataFrame({
        'neighbourhood_cleansed': [neighborhood_input],
        'property_type': [property_input],
        'room_type': [room_input],
        'accommodates': [accommodates],
        'bathrooms': [bathrooms],
        'beds': [beds],
        'nightly_price': [price],
        'review_scores_rating': [rating]
    })

    # Convert to type int or float
    input_df = input_df.astype({
        'accommodates': 'int',
        'bathrooms': 'float',
        'beds': 'int',
        'nightly_price': 'float',
        'review_scores_rating': 'float'
    })

    input_df.loc[0, 'neighbourhood_cleansed' ] = neighborhood_input
    input_df.loc[0, 'property_type'] = property_input
    input_df.loc[0, 'room_type' ] = room_input
    input_df.loc[0, 'accommodates'] = accommodates
    input_df.loc[0, 'bathrooms' ] = bathrooms
    input_df.loc[0, 'beds' ] = beds
    input_df.loc[0, 'nightly_price'] = price
    input_df.loc[0, 'review_scores_rating' ] = rating

    input_df = input_df.astype({"accommodates":"int",
                                "bathrooms":"float",
                                "beds":"int",
                                "nightly_price":"float",
                                "review_scores_rating":"float"})


    st.write((get_simplified_recommendations(sd_simplified_pp, input_df)))

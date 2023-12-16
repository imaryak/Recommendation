# import libraries 
import pandas as pd
import numpy as np
import math
from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt
from matplotlib import colorbar
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from pickle import dump, load

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

np.random_state = 42

sd_listings = pd.read_csv('data/sd_listings', index_col= 0)
# sd_listings.head(3)

modeling_cols = ['id', 'listing_url', 'latitude', 'longitude', 'neighbourhood_cleansed',
               'zipcode', 'property_type', 'room_type', 'accommodates', 
               'bathrooms', 'bedrooms', 'beds',
               'bed_type','nightly_price', 'price_per_stay',
               'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people',
               'minimum_nights', 'maximum_nights','host_response_time', 'host_response_rate',
               'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic',
               'host_identity_verified', 'number_of_reviews', 'number_of_stays',
               'review_scores_rating', 'review_scores_accuracy',
               'review_scores_cleanliness', 'review_scores_checkin',
               'review_scores_communication', 'review_scores_location',
               'review_scores_value', 'requires_license', 'instant_bookable',
               'is_business_travel_ready', 'cancellation_policy',
               'require_guest_profile_picture', 'require_guest_phone_verification']

sd_modeling = sd_listings[modeling_cols]


# print(sd_modeling.isna().sum())

# fill missing beds with values in bedrooms
sd_modeling['beds'] = sd_modeling['beds'].fillna(sd_modeling['bedrooms'])

sd_modeling[['security_deposit', 'cleaning_fee']] = sd_modeling[['security_deposit', 'cleaning_fee']].fillna(0)

sd_modeling['host_response_time'] = sd_modeling['host_response_time'].fillna('no response')

sd_modeling['host_response_time'].value_counts()

sd_modeling = sd_modeling.dropna()

sd_modeling.isna().sum()

# convert to appropriate datatype
sd_modeling['bedrooms'] = sd_modeling['bedrooms'].astype(float)
sd_modeling['zipcode'] = sd_modeling['zipcode'].astype(int)
sd_modeling['latitude'] = sd_modeling['latitude'].astype(float)
sd_modeling['longitude'] = sd_modeling['longitude'].astype(float)
sd_modeling['beds'] = sd_modeling['beds'].astype(float).astype(int)
sd_modeling['id'] = sd_modeling['id'].astype(int)

# sd_modeling[sd_modeling['zipcode']==92307]

# save as csv
path = "data/"

sd_modeling.to_csv(path + 'sd_modeling_with_urls')
sd_modeling = sd_modeling.drop(columns = 'listing_url')
# sd_modeling.head()


# save as csv
path = "data/"

sd_modeling.to_csv(path + 'sd_modeling_cleaned')

cat_cols = sd_modeling.select_dtypes(include = ['object']).columns
# cat_cols

for col in cat_cols:
    values = sd_modeling[col].value_counts()
    print(values, "\n")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25,20), sharey=True)

# Create bar plots
cat_cols = list(cat_cols)

for col, ax in zip(cat_cols, axes.flatten()):
    (sd_modeling.groupby(col)        # group values together by column of interest
         .count()['nightly_price']    # take the mean of the nightly_price for each group
         .sort_values()              # sort the groups in ascending order
         .plot
         .bar(ax=ax))                # create a bar graph on the ax
    
    ax.set_title(col)                # Make the title the name of the column
    
# fig.tight_layout()

sd_modeling.hist(figsize = (20,20))
plt.show()

# Set up figure size
fig, ax = plt.subplots(figsize=(25, 25))

# Select only numeric columns from the DataFrame
numeric_columns = sd_modeling.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix for numeric columns
corr = numeric_columns.corr()

# Plot the heatmap
sns.heatmap(corr, cmap='coolwarm', annot=True)

# Customize the plot appearance
ax.set_title("Heatmap of Correlation Between Features")

# Show the plot
plt.show()


sd_modeling['zipcode'] = sd_modeling['zipcode'].astype(object)


# convert to categorical dtype
sd_modeling['host_response_time'] = sd_modeling['host_response_time'].astype('category')
sd_modeling['cancellation_policy'] = sd_modeling['cancellation_policy'].astype('category')

# define order of the ordinal features
response_time_list = ['within an hour',
                      'within a few hours', 
                      'within a day', 
                      'a few days or more', 
                      'no response']

cancellation_policy_list = ['flexible',
                            'moderate',
                            'strict',
                            'strict_14_with_grace_period', 
                            'super_strict_60', 
                            'super_strict_30',
                            'luxury_moderate']

# define nominal and ordinal features in the categorical columns
nom_cols = sd_modeling.select_dtypes(['object']).columns
# print(nom_cols)
ordinal_cols = sd_modeling.select_dtypes(['category']).columns
# print(ordinal_cols)

# define numeric transformation pipeline that scales the numbers
numeric_pipeline = Pipeline([('numnorm', StandardScaler())]) 

# define an ordinal transformation pipeline that ordinal encodes the cats
ordinal_pipeline = Pipeline([('ordinalenc', OrdinalEncoder(categories = [response_time_list, 
                                                                         cancellation_policy_list]))])

# define a nominal transformation pipeline that OHE the cats
nominal_pipeline = Pipeline([('onehotenc', OneHotEncoder(categories= "auto", 
                                                         sparse = False, 
                                                         handle_unknown = 'ignore'))]) 

sd_trans = sd_modeling.drop(columns = ['id', 'latitude', 'longitude'])
sd_trans = sd_trans.reset_index(drop = 'index')
# sd_trans.head()    

# construct column transformer for the selected columns with pipelines
ct = ColumnTransformer(transformers = [("nominalpipe", nominal_pipeline, ['neighbourhood_cleansed',
                                                                            'zipcode', 'property_type', 
                                                                            'room_type','bed_type']),
                                       ("ordinalpipe", ordinal_pipeline, ['host_response_time', 
                                                                          'cancellation_policy']),
                                       ("numericpipe", numeric_pipeline, sd_trans.select_dtypes(['int', 'float']).columns)])
dump(ct, open('/notebooks/column_transformer.pkl', 'wb'))
# reorder the df with nominal first, ordinal second, and remaining numeric last

sd_trans = sd_trans[ # nominal below this line
                    ['neighbourhood_cleansed','zipcode', 'property_type', 'room_type','bed_type',
                     # ordinal below this line
                     'host_response_time', 'cancellation_policy', 
                     # remaining numeric
                     'accommodates', 'bathrooms', 'bedrooms', 'beds','nightly_price', 'price_per_stay', 
                     'security_deposit', 'cleaning_fee','guests_included', 'extra_people', 'minimum_nights', 
                     'maximum_nights', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 
                     'host_has_profile_pic','host_identity_verified', 'number_of_reviews', 'number_of_stays', 'review_scores_rating', 
                     'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
                     'review_scores_communication', 'review_scores_location', 'review_scores_value', 
                     'requires_license', 'instant_bookable', 'is_business_travel_ready',
                     'require_guest_profile_picture', 'require_guest_phone_verification']]

# sd_trans.head()

# sd_trans.shape

# save as csv
path = "data/"

sd_trans.to_csv(path + 'sd_trans')

sd_pp = pd.DataFrame(ct.fit_transform(sd_trans))

# sd_pp.head()

# Assuming 'nominal_pipeline' is your OneHotEncoder within a pipeline
nominal_encoder = nominal_pipeline.named_steps['onehotenc']

# Fit and transform the data with the encoder
transformed_data = nominal_encoder.fit_transform(sd_trans[nom_cols])

# Get feature names using get_feature_names_out
nominal_features = list(nominal_encoder.get_feature_names_out(nom_cols))

# Now nominal_features contains the feature names after one-hot encoding
# nominal_features = list(nominal_pipeline.named_steps['onehotenc'].fit(sd_trans[nom_cols]).get_feature_names())

# nominal_features[0:5]

ordinal_list = list(ordinal_cols)
# ordinal_list 

numeric_list = ['accommodates', 'bathrooms', 'bedrooms', 'beds','nightly_price', 'price_per_stay', 
                     'security_deposit', 'cleaning_fee','guests_included', 'extra_people', 'minimum_nights', 
                     'maximum_nights', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 
                     'host_has_profile_pic','host_identity_verified', 'number_of_reviews', 'number_of_stays', 'review_scores_rating', 
                     'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
                     'review_scores_communication', 'review_scores_location', 'review_scores_value', 
                     'requires_license', 'instant_bookable', 'is_business_travel_ready',
                     'require_guest_profile_picture', 'require_guest_phone_verification']
# len(numeric_list)

features_to_clean = ['x0_Allied Gardens', 'x0_Alta Vista', 'x0_Amphitheater And Water Park', 'x0_Balboa Park',
                   'x0_Bario Logan', 'x0_Bay Ho', 'x0_Bay Park', 'x0_Bay Terrace', 'x0_Bird Land',
                   'x0_Bonita Long Canyon', 'x0_Carmel Mountain', 'x0_Carmel Valley', 'x0_Chollas View',
                   'x0_City Heights East', 'x0_City Heights West', 'x0_Clairemont Mesa', 'x0_College Area',
                   'x0_Columbia', 'x0_Core', 'x0_Cortez Hill', 'x0_Darnall', 'x0_Del Cerro',
                   'x0_Del Mar Heights', 'x0_East Lake', 'x0_East Village', 'x0_Eastlake Trails',
                   'x0_Eastlake Vistas', 'x0_Eastlake Woods', 'x0_Egger Highlands', 'x0_El Cerritos',
                   'x0_Emerald Hills', 'x0_Encanto', 'x0_Estlake Greens', 'x0_Gaslamp Quarter', 'x0_Gateway',
                   'x0_Grant Hill', 'x0_Grantville', 'x0_Horton Plaza', 'x0_Jomacha-Lomita', 'x0_Kearny Mesa',
                   'x0_Kensington', 'x0_La Jolla', 'x0_La Jolla Village', 'x0_Lake Murray', 'x0_Lincoln Park',
                   'x0_Linda Vista', 'x0_Little Italy', 'x0_Loma Portal', 'x0_Lynwood Hills', 'x0_Marina',
                   'x0_Memorial', 'x0_Midtown', 'x0_Midtown District', 'x0_Mira Mesa', 'x0_Mission Bay',
                   'x0_Mission Valley', 'x0_Moreno Mission', 'x0_Mount Hope', 'x0_Mountain View', 'x0_Nestor',
                   'x0_Normal Heights', 'x0_North City', 'x0_North Clairemont', 'x0_North Hills',
                   'x0_Northwest', 'x0_Oak Park', 'x0_Ocean Beach', 'x0_Old Town', 'x0_Otay Ranch',
                   'x0_Pacific Beach', 'x0_Palm City', 'x0_Paradise Hills', 'x0_Park West',
                   'x0_Paseo Ranchoero', 'x0_Rancho Bernadino', 'x0_Rancho Del Rey', 'x0_Rancho Penasquitos',
                   'x0_Rolando', 'x0_Rolling Hills Ranch', 'x0_Roseville', 'x0_Sabre Springs', 'x0_San Carlos',
                   'x0_San Ysidro', 'x0_Scripps Ranch', 'x0_Serra Mesa', 'x0_Sky Line', 'x0_Sorrento Valley',
                   'x0_South Park', 'x0_Southcrest', 'x0_Southwest', 'x0_Sunbow', 'x0_Talmadge',
                   'x0_Terra Nova', 'x0_Thomy Locust Pl', 'x0_Tierrasanta', 'x0_Tijuana River Valley',
                   'x0_Torrey Pines', 'x0_University City', 'x0_Valencia Park', 'x0_Webster',
                   'x0_West University Heights', 'x0_Wooded Area', 'x0_Yosemite Dr', 'x1_22000', 'x1_22010',
                   'x1_22425', 'x1_22435', 'x1_91901', 'x1_91902', 'x1_91910', 'x1_91911', 'x1_91913',
                   'x1_91914', 'x1_91915', 'x1_91932', 'x1_91941', 'x1_91942', 'x1_91945', 'x1_91950',
                   'x1_92014', 'x1_92025', 'x1_92029', 'x1_92037', 'x1_92054', 'x1_92064', 'x1_92071',
                   'x1_92075', 'x1_92101', 'x1_92102', 'x1_92103', 'x1_92104', 'x1_92105', 'x1_92106',
                   'x1_92107', 'x1_92108', 'x1_92109', 'x1_92110', 'x1_92111', 'x1_92113', 'x1_92114',
                   'x1_92115', 'x1_92116', 'x1_92117', 'x1_92118', 'x1_92119', 'x1_92120', 'x1_92121',
                   'x1_92122', 'x1_92123', 'x1_92124', 'x1_92126', 'x1_92127', 'x1_92128', 'x1_92129',
                   'x1_92130', 'x1_92131', 'x1_92139', 'x1_92154', 'x1_92173', 'x1_92307', 'x1_92618',
                   'x1_921096', 'x2_Aparthotel', 'x2_Apartment', 'x2_Barn', 'x2_Bed and breakfast', 'x2_Boat',
                   'x2_Boutique hotel', 'x2_Bungalow', 'x2_Bus', 'x2_Cabin', 'x2_Camper/RV', 'x2_Campsite',
                   'x2_Casa particular (Cuba)', 'x2_Castle', 'x2_Cave', 'x2_Chalet', 'x2_Condominium',
                   'x2_Cottage', 'x2_Dome house', 'x2_Earth house', 'x2_Farm stay', 'x2_Guest suite',
                   'x2_Guesthouse', 'x2_Hostel', 'x2_Hotel', 'x2_House', 'x2_Igloo', 'x2_Loft',
                   'x2_Nature lodge', 'x2_Other', 'x2_Resort', 'x2_Serviced apartment', 'x2_Tent',
                   'x2_Tiny house', 'x2_Townhouse', 'x2_Treehouse', 'x2_Vacation home', 'x2_Villa',
                   'x3_Entire home/apt', 'x3_Private room', 'x3_Shared room', 'x4_Airbed', 'x4_Couch',
                   'x4_Futon', 'x4_Pull-out Sofa', 'x4_Real Bed']


# removes the OHE strings at front end of feature names
def clean_features(lst):
    new_list = []
    for value in lst:
        splitted = value.split('_')[1] # returns name of feature after '_'
        new_list.append(splitted)
        continue
    return new_list                   

cleaned_features = clean_features(features_to_clean)

sd_pp.columns = cleaned_features + ordinal_list + numeric_list
# sd_pp

# save as csv
path = "data/"

sd_pp.to_csv(path + 'sd_pp')


# import umap

embedding = np.load('data/embedding_plot.npy')

fig, ax = plt.subplots(figsize=(12, 12))
sns.scatterplot(x=embedding.T[0], y=embedding.T[1], s=5, alpha=1)
plt.show()

sns.jointplot(x=embedding.T[0], y=embedding.T[1], kind='hex', height=12)
plt.show()

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

cluster_results = pd.read_csv('data/cluster_results', index_col= 0)

clusters = 5

fig, ax = plt.subplots()
ax.set_title('Inertia vs. Number of Clusters (2 - 40)')
cluster_results.inertia.plot(ax=ax)
ax.vlines(x=clusters, ymin=100000, ymax=500000, colors='red', linestyles='dotted')
plt.show()

fig, ax = plt.subplots()
ax.set_title('Silhouette Score vs. Number of Clusters (2 - 40)')
cluster_results.silhouette.plot()
ax.vlines(x=clusters, ymin=-.1, ymax=.6, colors='red', linestyles='dotted')
plt.show()

kmeans = MiniBatchKMeans(n_clusters = clusters, random_state = 42).fit(sd_pp)
# kmeans

# labels for each cluster
pred_labels = np.unique(kmeans.predict(sd_pp))
# pred_labels

# get prediction on sd_pp to get labels
sd_cluster_labels = pd.Series(kmeans.predict(sd_pp), index = sd_pp.index)

# convert predicted cluster labels to a list
cluster_list = sd_cluster_labels.values.tolist()

# insert cluster label to the sd_modeling df
sd_clustered = sd_modeling.copy()
sd_clustered.insert(1, 'cluster_label', cluster_list)

# sd_clustered.head()

# save as csv
path = "data/"
sd_clustered.to_csv(path + 'sd_clustered')

# insert cluster label to the sd_modeling df
sd_pp_clustered = sd_pp.copy()
sd_pp_clustered.insert(0, 'cluster_label', cluster_list)

# sd_pp_clustered.head()

# plot 6 clusters with UMAP 
fig, ax = plt.subplots(figsize = (15,12))

plt.scatter(*embedding.T, s = 5, alpha = 1, c = kmeans.predict(sd_pp), cmap = 'Spectral')

labels = sorted(list(sd_cluster_labels.unique())) # cluster labels

cbar = plt.colorbar(boundaries=np.arange(6)-0.5) # 6 cluster labels
cbar.set_ticks(np.arange(5))
cbar.set_ticklabels(labels)
plt.title('San Diego Airbnb Listings Embedded via UMAP');

plt.show()
import streamlit as st
st.title("Clustering")

st.pyplot(fig)



# embedding_map = umap.UMAP(n_neighbors=500, min_dist= .9, random_state=42).fit(sd_pp)

sd_pp_clustered.head(2)

# assigns cluster labels
cluster_df = pd.DataFrame(sd_pp_clustered['cluster_label']) 

# assigns neighborhood
neighborhood_df = pd.DataFrame(sd_pp_clustered.loc[:,'Allied Gardens':'Yosemite Dr'].idxmax(axis = 1)).rename(columns = {0:'Neighborhood'})

# assigns property type
property_df = pd.DataFrame(sd_pp_clustered.loc[:,'Aparthotel':'Villa'].idxmax(axis = 1)).rename(columns = {0:'Property'})

# assigns room type
room_df = pd.DataFrame(sd_pp_clustered.loc[:,'Entire home/apt':'Shared room'].idxmax(axis = 1)).rename(columns = {0:'Room Type'})

# assigns price
price = pd.DataFrame('$' + sd_trans['price_per_stay'].astype(str))

# assigns rating
rating = pd.DataFrame(sd_trans['review_scores_rating'].astype(int))


# concat all the dfs
cluster_hover_data = pd.concat([cluster_df, price, rating, neighborhood_df, property_df, room_df], axis = 1)

# cluster_hover_data

kmeans.predict(sd_pp)

# # p = umap.plot.interactive(embedding_map, 
#                           labels = kmeans.predict(sd_pp), # color codes the clusters
#                           point_size=5, alpha = 0.3,
#                           hover_data=cluster_hover_data, # get cluster labels
#                           width = 800, height = 800)
# umap.plot.show(p)
import streamlit as st
st.title("Viewing the cluster data")
# st.plotly_chart(p)


# cleaning up columns to plot 
cluster_hover_data['price_per_stay'] = cluster_hover_data['price_per_stay'].str.replace('$', '')
cluster_hover_data['price_per_stay'] = cluster_hover_data['price_per_stay'].astype(float)
cluster_hover_data['cluster_label'] = cluster_hover_data['cluster_label'].astype('category')
cluster_hover_data

# save as csv
path = "data/"

cluster_hover_data.to_csv(path + 'cluster_hover_data')

fig, axes = plt.subplots(figsize=(12,8))

ax = sns.histplot(cluster_hover_data['cluster_label'])

# set bar colors
ax.patches[0].set_facecolor('#A93226')
ax.patches[1].set_facecolor('#E67E22')
ax.patches[2].set_facecolor('#ECF87F')
ax.patches[3].set_facecolor('#52BE80')
ax.patches[4].set_facecolor('#A569BD')

plt.show()

palette = {0: '#A93226', 1:'#E67E22', 2:'#ECF87F', 3:'#52BE80', 4:'#A569BD'}

fig, axes = plt.subplots(figsize=(10,15))

ax = sns.histplot(data=cluster_hover_data, 
             x='Room Type', 
             hue='cluster_label', palette = palette,
             multiple="stack").set(title='Room Type Grouped By Cluster Label')

plt.show()

fig, axes = plt.subplots(figsize=(15,8))

# create a filter of greater than 70 rating to remove outliers
greater_than_70 = cluster_hover_data[cluster_hover_data['review_scores_rating'] >= 70]

ax = sns.histplot(data=greater_than_70, 
             x='review_scores_rating', 
             hue='cluster_label', palette = palette,
             multiple="stack").set(title='Average Review Rating (>=70) Grouped By Cluster Label')

plt.show()

fig, axes = plt.subplots(figsize=(15,8))

# create a filter of LESS than 70 rating to remove outliers
less_than_70 = cluster_hover_data[cluster_hover_data['review_scores_rating'] <= 70]

sns.histplot(data=less_than_70, 
             x='review_scores_rating', 
             hue='cluster_label', palette = palette, 
             multiple="stack").set(title='Average Review Rating (<=70) Grouped By Cluster Label')

plt.show()

fig, axes = plt.subplots(figsize=(12, 8))

# create a filter of less than $1000 for price per stay to remove outliers
less_than_2000 = cluster_hover_data[cluster_hover_data['price_per_stay'] <= 1000]

# Generate a dynamic palette based on unique values in the 'cluster_label' column
unique_clusters = less_than_2000['cluster_label'].unique()
palette = sns.color_palette("husl", n_colors=len(unique_clusters))

sns.boxplot(
    x=less_than_2000['cluster_label'],
    y=less_than_2000['price_per_stay'],
    palette=palette
).set(title='Price Per Stay (<$1000) Grouped By Cluster Label')

plt.show()

cluster_hover_data.groupby(['Property', 'cluster_label']).size().unstack(fill_value=0).plot.barh(figsize=(10,18), 
                                                                                                 color = palette)

plt.title('Properties by Cluster Group')
plt.show()


import streamlit as st

# Display the conclusions about the clusters
st.title("Conclusions About the Clusters")

# Cluster Label 0
st.header("Cluster Label 0 (Red) - Favorable high-end listings")
st.markdown(
    "- Favorable and wide range of review ratings.\n"
    "- Most expensive listings and mostly consist of entire home room types."
)

# Cluster Label 1
st.header("Cluster Label 1 (Orange) - Favorable highly rated & moderately priced listings")
st.markdown(
    "- Popular group, generally > 90 review ratings, relatively inexpensive.\n"
    "- Mostly houses or private rooms, wide range of property types."
)

# Cluster Label 2
st.header("Cluster Label 2 (Yellow) - Favorable moderately priced diverse listings")
st.markdown(
    "- Most popular group, mostly favorable ratings.\n"
    "- Relatively low priced. Wide range of property types."
)

# Cluster Label 3
st.header("Cluster Label 3 (Green) - Favorable and least expensive listings")
st.markdown(
    "- Popular group and wide range of review ratings.\n"
    "- Least expensive group. Wide range of property types."
)

# Cluster Label 4
st.header("Cluster Label 4 (Purple) - Unfavorable listings")
st.markdown(
    "- Least popular group and lowest-rated listings."
)

# Note: Customize the content as needed for your specific conclusions.

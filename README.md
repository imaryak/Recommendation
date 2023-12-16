# FinalProject

[ CodeLab Link](https://docs.google.com/document/d/17qjfYyA1okO8cFkS3EaMT7IfeTyvko7eD6KR_OgM9UU/edit?usp=sharing
)
# Airbnb Listing Analysis Project

## Overview

This project involves the analysis of Airbnb listings, combining machine learning techniques to offer personalized recommendations, cluster insights, and NLP-driven textual analysis. The primary goals include providing users with tailored recommendations, identifying common characteristics within distinct listing groups, and gaining insights into how hosts describe their listings.

## Machine Learning Concepts Employed

### 1. Recommendation System

#### Objective
•⁠  ⁠Recommend listings similar to those users have previously enjoyed.

#### Technical Implementation
•⁠  ⁠User-based content filtering using cosine similarity.
•⁠  ⁠Dataset transformed into a matrix, with cosine similarity calculated between the selected listing and others.

### 2. Cluster Analysis

#### Objective
•⁠  ⁠Visualize and understand patterns in high-dimensional data.
•⁠  ⁠Group listings into distinct clusters based on similar features.

#### Technical Implementation
•⁠  ⁠UMAP dimensionality reduction for visualization.
•⁠  ⁠MiniBatch KMeans clustering for grouping listings.

### 3. Natural Language Processing (NLP) Analysis

#### Text Data Cleaning

•⁠  ⁠Objective: Prepare text data for analysis by normalizing, tokenizing, and lemmatizing.
•⁠  ⁠Technical Implementation: Removal of stop words, lemmatization, and tokenization.

#### Sentiment Analysis

•⁠  ⁠Objective: Understand sentiment in textual descriptions.
•⁠  ⁠Technical Implementation: Sentiment analysis using the TextBlob module, providing polarity and subjectivity scores.

## Use Cases

### Personalized Recommendations

•⁠  ⁠Objective: Offer users tailored recommendations based on their past stays.
•⁠  ⁠Technical Implementation: Leveraging cosine similarity scores to rank and recommend listings.

### Cluster Analysis Insights

•⁠  ⁠Objective: Identify common characteristics within distinct groups of Airbnb listings.
•⁠  ⁠Technical Implementation: Exploring features prevalent in each cluster.

### NLP Analysis for Textual Descriptions

•⁠  ⁠Objective: Gain insights into how hosts typically describe their listings.
•⁠  ⁠Technical Implementation: Generating word clouds to visually represent term frequency.

## Implementation Details

•⁠  ⁠*Programming Language:* Python
•⁠  ⁠*Libraries Used:* pandas, numpy, scikit-learn, NLTK, spaCy, TextBlob, Streamlit, and others.
•⁠  ⁠*Machine Learning Models:* Cosine similarity, UMAP dimensionality reduction, MiniBatch KMeans clustering.
•⁠  ⁠*Web Application:* Built using Streamlit for user interaction.

## How to Use

 1.⁠ ⁠Clone the repository.
 2.⁠ ⁠Install required dependencies using ⁠ pip install -r requirements.txt ⁠.
 3.⁠ ⁠Run the Streamlit application with ⁠ streamlit run airbnb_algorithmic_marketing_file.py ⁠.

import streamlit as st
from textblob import TextBlob

# Function to perform sentiment analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    # Classify the polarity of the sentiment (positive, negative, or neutral)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit app
def main():
    st.title("Hotel Review Sentiment Analysis")
    
    # Text input for user to enter hotel review
    review = st.text_area("Enter your hotel review:")

    if st.button("Analyze Sentiment"):
        if review:
            # Perform sentiment analysis
            sentiment = analyze_sentiment(review)
            
            # Display result
            st.write(f"Sentiment: {sentiment}")
        else:
            st.warning("Please enter a hotel review.")

if __name__ == "__main__":
    main()

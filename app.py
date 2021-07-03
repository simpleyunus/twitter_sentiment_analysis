import streamlit as st
import joblib 
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# loading model
rnd_clf = joblib.load('model.pkl')

analyzer = SentimentIntensityAnalyzer()



html_temp = """
<div style = "background-color:tomato;padding:10px">
<h2 style = "color:white;text-align:center;">Zimbabwe Politics Sentiment Analysis From Twitter Data</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)


st.header(" Mood Changes From December to June")

read = pd.read_csv('mymoods.csv')
moods = pd.DataFrame(read)
moods = moods.drop('Unnamed: 0', axis=1)

st.write(moods)
st.line_chart(moods)
st.write('Negative : 0')
st.write('Neutral : 1')
st.write('Positive : 2')

st.info("Prediction Application Of A Tweet")

tweet = st.text_input("Enter Text to determine Polarity")
button = st.button("Predict")

text = pd.DataFrame(pd.read_csv('mydata.csv'))
text = text.drop('Unnamed: 0', axis=1)
te = text['clean_tweets']




def predict(text, tweet):
    text[0] = tweet
    bow_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
    vectorized_tweets = bow_vectorizer.fit_transform(text.values.astype('str'))
    result = rnd_clf.predict(vectorized_tweets)
    return result

if button:
    tr = tweet
    pol = predict(te, tr)
    st.write(pol[0])
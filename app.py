import streamlit as st
import joblib 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# loading model
rnd_clf = joblib.load('model.pkl')

st.title("Sentiment Analysis Using Twitter Data")
st.header("Displaying Mood Changes Since December to June")

read = pd.read_csv('mymoods.csv')
moods = pd.DataFrame(read)
moods = moods.drop('Unnamed: 0', axis=1)

st.write(moods)
st.line_chart(moods)
st.write('Negative : 0')
st.write('Neutral : 1')
st.write('Positive : 2')

st.info("The App will use the loaded model to predict the polarity of a tweet")

tweet = st.text_input("Enter Text to determine Polarity")
button = st.button("Predict")

text = pd.DataFrame(pd.read_csv('mydata.csv'))
text = text.drop('Unnamed: 0', axis=1)
te = text['clean_tweets']
# rec = "me"
# te[0] = rec
# st.write(te[0])



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
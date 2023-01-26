import pandas as pd
import regex
import string
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import joblib

model=joblib.load('TweetPred.pkl')

st.title('Disaustruos Tweet Detection')
text=st.text_input('Enter your text')

pred=1
if st.button('Check'):
    # text.replace("[^a-zA-Z]", " ",regex = True, inplace = True)
    text.lower()
    # text=[x for x in text.split() if x not in string.punctuation]
    vectorizer = TfidfVectorizer()
    text = vectorizer.fit_transform(text)
    pred=model.predict(text)
st.write(pred)

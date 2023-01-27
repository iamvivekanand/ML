import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import joblib
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
model=joblib.load('mymodel.pkl')

# model saving
st.title('Email Spam Detection')
sample=st.text_input("Enter your Email Headline or Subject:")
# st.write(sample)
res=''
if st.button('Detect'):
    sample=cv.transform([sample]).toarray()
    res=model.predict(sample)
    if res=='ham':
        st.success('Not Spam')
    else:
        st.write('Tips: mails offering cash, prizes, high paid jobs are likely to be a spam')
        st.error('Spam ')
st.success('This model is Developed by Vivekanand')


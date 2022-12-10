import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 
df=pd.read_csv('fake_or_real_news.csv')
# df.head(2)
# df.shape
df['label'].unique()
len(df['Unnamed: 0'].unique())
# 
X=np.array(df['title'])
y=np.array(df['label'])
# 
cv=CountVectorizer()
X=cv.fit_transform(X)
# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# 
model=MultinomialNB()
model.fit(X_train,y_train)

import streamlit as st
st.title('Fake News Detection System')
text=st.text_input('Enter your news Headline: ')
if st.button('Predict'):
    if len(text)<1:
        st.error('Pleae Enter A Valid Text')
    else:
        text=cv.transform([text]).toarray()
        res=model.predict(text)
        if res=='fake':
            st.error('Fake News')
        else:
            st.success('Not Fake')

st.write('Developed by Vivekanand')
# def fakenewsdetect():
    # usertext=st.text_area("enter any news headline:")
    # if len(usertext)<1:
        # st.write(" ")
    # else:
        # usertext=cv.transform([usertext]).toarray()
        # res=model.predict(usertext)
        # st.title(res)
# fakenewsdetect()

import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("spam.csv", encoding= 'latin-1')
df=data.copy()
df=df[['v1','v2']]
df.columns=['target','emails']
X=np.array(df['emails'])
y=np.array(df['target'])
X=cv.fit_transform(X)
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
model = MultinomialNB()
model.fit(X_train,y_train)

# model=joblib.load('mymodel.pkl')

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


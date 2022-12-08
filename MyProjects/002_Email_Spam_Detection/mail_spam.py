import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
# 
data = pd.read_csv("spam.csv", encoding= 'latin-1')
df=data.copy()
# rename columns for convenience
df=df[['v1','v2']]
df.columns=['target','emails']
df.head(3)
# 
X=np.array(df['emails'])
y=np.array(df['target'])
# 
cv=CountVectorizer()
X=cv.fit_transform(X)
# 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# 
model = MultinomialNB()
model.fit(X_train,y_train)
# 
# sample=input('enter a message:')
# sample=cv.transform([sample]).toarray()
# model.predict(sample)

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
st.warning('This model is Developed by Vivekanand')



